import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import sys
import traceback
from typing import Any, Dict, Optional

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler


# -----------------------------
# Repro helpers
# -----------------------------
def _now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_json_dump(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _try_set_task_seed(task: Any, seed: int) -> None:
    for name in ["set_seed", "seed_everything", "reset_seed"]:
        fn = getattr(task, name, None)
        if callable(fn):
            try:
                fn(seed)
                return
            except Exception:
                pass

    for attr in ["seed", "random_seed"]:
        if hasattr(task, attr):
            try:
                setattr(task, attr, seed)
                return
            except Exception:
                pass


def _make_task_with_seed(seed: int,args) -> Any:
    try:
        # task = OBPEvaluation(seed=seed)
        task = OBPEvaluation(seed=seed, instance_seed=args.instance_seed)
    except TypeError:
        task = OBPEvaluation()
        _try_set_task_seed(task, seed)
    return task


def _safe_get_best_score(method: Any) -> Optional[float]:
    for attr in ["best_score", "best_fitness", "best_value", "global_best_score"]:
        v = getattr(method, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _safe_evaluate(task: Any, program: Any) -> Optional[float]:
    prog = program if isinstance(program, str) else str(program)
    for name in ["evaluate", "eval", "run", "__call__"]:
        fn = getattr(task, name, None)
        if callable(fn):
            try:
                out = fn(prog)
                if isinstance(out, (int, float)):
                    return float(out)
                if isinstance(out, dict):
                    for k in ["score", "best_score", "fitness", "value"]:
                        if k in out and isinstance(out[k], (int, float)):
                            return float(out[k])
                if isinstance(out, (tuple, list)) and out and isinstance(out[0], (int, float)):
                    return float(out[0])
            except Exception:
                continue
    return None


def _make_llm(host: str, model: str, api_key: str, timeout: int,
              temperature: Optional[float], top_p: Optional[float]) -> HttpsApi:
    kwargs = dict(host=host, key=api_key, model=model, timeout=timeout)
    extra = {}
    if temperature is not None:
        extra["temperature"] = temperature
    if top_p is not None:
        extra["top_p"] = top_p
    try:
        return HttpsApi(**kwargs, **extra)
    except TypeError:
        return HttpsApi(**kwargs)


def write_summary_csv(path: str, rows: list) -> None:
    fieldnames = [
        "backend",
        "seed",
        "ok",
        "best_score",
        "best_program_sha256",
        "best_program_path",
        "time_start",
        "time_end",
        "error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


# -----------------------------
# Main experiment runner
# -----------------------------
def run_one_seed(
    backend_name: str,
    host: str,
    model: str,
    api_key: str,
    args: argparse.Namespace,
    backend_run_dir: str,
    seed: int,
) -> Dict[str, Any]:
    seed_dir = os.path.join(backend_run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    log_dir = os.path.join(seed_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    _set_global_seed(seed)

    task = _make_task_with_seed(seed,args)
    llm = _make_llm(
        host=host,
        model=model,
        api_key=api_key,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    profiler = EoHProfiler(log_dir=log_dir, log_style="simple")

    method = EoH(
        llm=llm,
        evaluation=task,
        profiler=profiler,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        max_sample_nums=args.max_sample_nums,
        selection_num=args.selection_num,
        use_e2_operator=args.use_e2_operator,
        use_m1_operator=args.use_m1_operator,
        use_m2_operator=args.use_m2_operator,
        num_samplers=args.num_samplers,
        num_evaluators=args.num_evaluators,
        debug_mode=args.debug_mode,
    )

    seed_cfg = {
        "backend": backend_name,
        "seed": seed,
        "time_start": _now_iso(),
        "host": host,
        "model": model,
        "timeout": args.timeout,
        "eoh": {
            "pop_size": args.pop_size,
            "max_generations": args.max_generations,
            "max_sample_nums": args.max_sample_nums,
            "selection_num": args.selection_num,
            "use_e2_operator": args.use_e2_operator,
            "use_m1_operator": args.use_m1_operator,
            "use_m2_operator": args.use_m2_operator,
            "num_samplers": args.num_samplers,
            "num_evaluators": args.num_evaluators,
            "debug_mode": args.debug_mode,
        },
    }
    _safe_json_dump(os.path.join(seed_dir, "config.json"), seed_cfg)

    result: Dict[str, Any] = {
        "backend": backend_name,
        "seed": seed,
        "ok": False,
        "best_score": None,
        "best_program_sha256": None,
        "best_program_path": None,
        "error": None,
        "time_start": seed_cfg["time_start"],
        "time_end": None,
    }

    try:
        print(f"[{backend_name}][seed={seed}] >>> Start EoH on OBP ...")
        best_program = method.run()
        print(f"[{backend_name}][seed={seed}] >>> Search finished.")

        best_program_text = best_program if isinstance(best_program, str) else str(best_program)
        best_path = os.path.join(seed_dir, "best_program.txt")
        _write_text(best_path, best_program_text)

        sha = _sha256_text(best_program_text)
        _write_text(os.path.join(seed_dir, "best_program.sha256"), sha)

        best_score = _safe_get_best_score(method)
        if best_score is None:
            best_score = _safe_evaluate(task, best_program)

        result.update(
            {
                "ok": True,
                "best_score": best_score,
                "best_program_sha256": sha,
                "best_program_path": os.path.abspath(best_path),
            }
        )

    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        result["error"] = err
        _write_text(os.path.join(seed_dir, "error.txt"), err)
        print(f"[{backend_name}][seed={seed}] !!! ERROR:\n{err}")

    result["time_end"] = _now_iso()
    _safe_json_dump(os.path.join(seed_dir, "result.json"), result)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EoH OBP reproduction for 3 backends (DeepSeek/Qwen/Doubao).")

    # ===== backends =====
    # 注意：host 不要带 https://
    p.add_argument("--deepseek_host", type=str, default="api.deepseek.com")
    p.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    p.add_argument("--deepseek_key_env", type=str, default="DEEPSEEK_API_KEY")

    p.add_argument("--qwen_host", type=str, default="dashscope.aliyuncs.com")
    p.add_argument("--qwen_model", type=str, default="qwen-flash")
    p.add_argument("--qwen_key_env", type=str, default="QWEN_API_KEY")

    p.add_argument("--doubao_host", type=str, default="ark.cn-beijing.volces.com")
    p.add_argument("--doubao_model", type=str, default="doubao-seed-2-0-mini-260215")
    p.add_argument("--doubao_key_env", type=str, default="ARK_API_KEY")

    # ===== shared llm args =====
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)

    # ===== EoH params =====
    p.add_argument("--pop_size", type=int, default=10)
    p.add_argument("--max_generations", type=int, default=6)
    p.add_argument("--max_sample_nums", type=int, default=10000)
    p.add_argument("--selection_num", type=int, default=3)
    p.add_argument("--use_e2_operator", action="store_true", default=True)
    p.add_argument("--use_m1_operator", action="store_true", default=True)
    p.add_argument("--use_m2_operator", action="store_true", default=True)
    p.add_argument("--num_samplers", type=int, default=4)
    p.add_argument("--num_evaluators", type=int, default=4)
    p.add_argument("--debug_mode", action="store_true", default=False)

    # ===== Repro control =====
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out_dir", type=str, default="results/eoh_obp_repro_3")
    p.add_argument("--exp_name", type=str, default="eoh_obp_single_3backends")

    p.add_argument("--instance_seed", type=int, default=2026)
    return p.parse_args()


def _get_env_or_fail(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"API key not found. Please set environment variable {name}.")
    return v


def main() -> None:
    args = parse_args()

    # run root
    run_id = f"{args.exp_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # run-level meta
    run_meta = {
        "run_id": run_id,
        "time_start": _now_iso(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "args": {k: getattr(args, k) for k in vars(args)},
        "backends": [
            {"name": "deepseek", "host": args.deepseek_host, "model": args.deepseek_model, "key_env": args.deepseek_key_env},
            {"name": "qwen",     "host": args.qwen_host,     "model": args.qwen_model,     "key_env": args.qwen_key_env},
            {"name": "doubao",   "host": args.doubao_host,   "model": args.doubao_model,   "key_env": args.doubao_key_env},
        ],
    }
    _safe_json_dump(os.path.join(run_dir, "run_meta.json"), run_meta)

    # resolve keys
    deepseek_key = _get_env_or_fail(args.deepseek_key_env)
    qwen_key = _get_env_or_fail(args.qwen_key_env)
    doubao_key = _get_env_or_fail(args.doubao_key_env)

    backends = [
        ("deepseek", args.deepseek_host, args.deepseek_model, deepseek_key),
        ("qwen", args.qwen_host, args.qwen_model, qwen_key),
        ("doubao", args.doubao_host, args.doubao_model, doubao_key),
    ]

    all_rows = []

    for backend_name, host, model, api_key in backends:
        backend_run_dir = os.path.join(run_dir, backend_name)
        os.makedirs(backend_run_dir, exist_ok=True)

        backend_rows = []
        for seed in args.seeds:
            row = run_one_seed(
                backend_name=backend_name,
                host=host,
                model=model,
                api_key=api_key,
                args=args,
                backend_run_dir=backend_run_dir,
                seed=seed,
            )
            backend_rows.append(row)
            all_rows.append(row)

        # per-backend summary
        write_summary_csv(os.path.join(backend_run_dir, "summary.csv"), backend_rows)

    # global summary
    write_summary_csv(os.path.join(run_dir, "summary_all.csv"), all_rows)

    run_meta["time_end"] = _now_iso()
    _safe_json_dump(os.path.join(run_dir, "run_meta.json"), run_meta)

    print(f"\n=== DONE. All summaries saved under: {os.path.abspath(run_dir)} ===")
    print(f" - per-backend: */summary.csv")
    print(f" - global:     summary_all.csv")


if __name__ == "__main__":
    main()
    #python run_eoh_obp_repro.py --seeds 0 1 2 3 4