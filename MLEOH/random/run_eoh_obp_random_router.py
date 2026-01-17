# run_eoh_obp_random_router.py
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

from random_router import RandomRouter


# python run_eoh_obp_random_router.py --seeds 0 1 2 3 4 --pop_size 10 --max_generations 6 --max_sample_nums 10000 --p_model_a 0.5 --exp_name eoh_obp_random_router

# -----------------------------
# Repro helpers (同 run_eoh_obp_repro.py 风格)
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
        import numpy as np  # optional
        np.random.seed(seed)
    except Exception:
        pass


def _try_set_task_seed(task: Any, seed: int) -> None:
    # 1) 可能的 set_seed / seed 属性
    for name in ["set_seed", "seed_everything", "reset_seed"]:
        fn = getattr(task, name, None)
        if callable(fn):
            try:
                fn(seed)
                return
            except Exception:
                pass

    # 2) 直接设置属性
    for attr in ["seed", "random_seed"]:
        if hasattr(task, attr):
            try:
                setattr(task, attr, seed)
                return
            except Exception:
                pass


def _make_task_with_seed(seed: int) -> Any:
    try:
        task = OBPEvaluation(seed=seed)
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
    prog = program
    if not isinstance(prog, str):
        try:
            prog = str(program)
        except Exception:
            prog = None
    if prog is None:
        return None

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
                if isinstance(out, (tuple, list)) and out:
                    if isinstance(out[0], (int, float)):
                        return float(out[0])
            except Exception:
                continue
    return None


def _make_llm(
    host: str,
    model: str,
    api_key: str,
    timeout: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> HttpsApi:
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


# -----------------------------
# Main experiment runner
# -----------------------------
def run_one_seed(args: argparse.Namespace, run_dir: str, seed: int) -> Dict[str, Any]:
    seed_dir = os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    log_dir = os.path.join(seed_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    _set_global_seed(seed)

    # task
    task = _make_task_with_seed(seed)

    # backend LLMs
    llm_a = _make_llm(
        host=args.host_a,
        model=args.model_a,
        api_key=args.api_key_a,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    llm_b = _make_llm(
        host=args.host_b,
        model=args.model_b,
        api_key=args.api_key_b,
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # router seed：默认跟随 seed（方便复现），也允许显式指定一个固定 router_seed
    router_seed = seed if args.router_seed is None else int(args.router_seed)

    router_llm = RandomRouter(
        deepseek_llm=llm_a,
        qwen_llm=llm_b,
        p_model_a=args.p_model_a,
        seed=router_seed,
        deterministic_by_prompt=args.deterministic_by_prompt,
        verbose=args.router_verbose,
    )

    profiler = EoHProfiler(
        log_dir=log_dir,
        log_style="simple",
    )

    method = EoH(
        llm=router_llm,
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

    # 保存 seed 级配置（保持与你的单模型复现脚本一致风格，并扩展 router/backend）
    seed_cfg = {
        "seed": seed,
        "time_start": _now_iso(),
        "router": {
            "type": "random",
            "p_model_a": args.p_model_a,
            "router_seed": router_seed,
            "deterministic_by_prompt": args.deterministic_by_prompt,
        },
        "backend_a": {"host": args.host_a, "model": args.model_a, "api_key_env": args.api_key_env_a},
        "backend_b": {"host": args.host_b, "model": args.model_b, "api_key_env": args.api_key_env_b},
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

    # 结果（核心字段不变：与 run_eoh_obp_repro.py 对齐）
    result: Dict[str, Any] = {
        "seed": seed,
        "ok": False,
        "best_score": None,
        "best_program_sha256": None,
        "best_program_path": None,
        "error": None,
        "time_start": seed_cfg["time_start"],
        "time_end": None,
        # 扩展字段（不影响 summary.csv 的核心列）
        "router_type": "random",
        "router_seed": router_seed,
        "p_model_a": args.p_model_a,
        "deterministic_by_prompt": args.deterministic_by_prompt,
        "router_stats": None,
    }

    try:
        print(f"[seed={seed}] >>> Start EoH (RandomRouter) on Online Bin Packing ...")
        best_program = method.run()
        print(f"[seed={seed}] >>> Search finished.")

        best_program_text = best_program if isinstance(best_program, str) else str(best_program)
        best_path = os.path.join(seed_dir, "best_program.txt")
        _write_text(best_path, best_program_text)
        sha = _sha256_text(best_program_text)
        _write_text(os.path.join(seed_dir, "best_program.sha256"), sha)

        best_score = _safe_get_best_score(method)
        if best_score is None:
            best_score = _safe_evaluate(task, best_program)

        # router 统计
        router_stats = {
            "call_cnt": getattr(router_llm, "call_cnt", None),
            "model_a_cnt": getattr(router_llm, "deepseek_cnt", None),
            "model_b_cnt": getattr(router_llm, "qwen_cnt", None),
        }

        result.update(
            {
                "ok": True,
                "best_score": best_score,
                "best_program_sha256": sha,
                "best_program_path": os.path.abspath(best_path),
                "router_stats": router_stats,
            }
        )

    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        result["error"] = err
        _write_text(os.path.join(seed_dir, "error.txt"), err)
        print(f"[seed={seed}] !!! ERROR:\n{err}")

    finally:
        # 结束时间与落盘
        result["time_end"] = _now_iso()
        _safe_json_dump(os.path.join(seed_dir, "result.json"), result)

        # 关闭资源（router.close 会顺带 close 两个 backend）
        try:
            if hasattr(router_llm, "close"):
                router_llm.close()
        except Exception:
            pass

    return result


def write_summary_csv(path: str, rows: list) -> None:
    # 核心列：完全沿用 run_eoh_obp_repro.py 的列名与顺序
    fieldnames = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproducible EoH runner with RandomRouter (OBP).")

    # Backend A (默认 DeepSeek)
    p.add_argument("--host_a", type=str, default="api.deepseek.com")
    p.add_argument("--model_a", type=str, default="deepseek-chat")
    p.add_argument("--api_key_env_a", type=str, default="DEEPSEEK_API_KEY")

    # Backend B (默认 Qwen)
    p.add_argument("--host_b", type=str, default="dashscope.aliyuncs.com")
    p.add_argument("--model_b", type=str, default="qwen-flash")
    p.add_argument("--api_key_env_b", type=str, default="QWEN_API_KEY")

    # Common LLM params
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)

    # RandomRouter params
    p.add_argument("--p_model_a", type=float, default=0.5, help="Probability to choose backend A.")
    p.add_argument("--router_seed", type=int, default=None, help="If None, use the same as seed.")
    p.add_argument("--deterministic_by_prompt", action="store_true", default=True,
                   help="Make routing deterministic as hash(seed,prompt), robust to parallelism.")
    p.add_argument("--router_verbose", action="store_true", default=False)

    # EoH params
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

    # Repro control (same style)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--out_dir", type=str, default="results/eoh_obp_repro")
    p.add_argument("--exp_name", type=str, default="eoh_obp_random_router")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load API keys from env
    api_key_a = os.environ.get(args.api_key_env_a, "").strip()
    api_key_b = os.environ.get(args.api_key_env_b, "").strip()
    if not api_key_a:
        raise RuntimeError(f"API key A not found. Please set env var {args.api_key_env_a}.")
    if not api_key_b:
        raise RuntimeError(f"API key B not found. Please set env var {args.api_key_env_b}.")

    args.api_key_a = api_key_a
    args.api_key_b = api_key_b

    run_id = f"{args.exp_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # run-level meta（同风格）
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
        "args": {k: getattr(args, k) for k in vars(args) if not k.startswith("api_key_")},
    }
    _safe_json_dump(os.path.join(run_dir, "run_meta.json"), run_meta)

    all_rows = []
    for seed in args.seeds:
        row = run_one_seed(args=args, run_dir=run_dir, seed=seed)
        all_rows.append(row)

    summary_path = os.path.join(run_dir, "summary.csv")
    write_summary_csv(summary_path, all_rows)

    run_meta["time_end"] = _now_iso()
    _safe_json_dump(os.path.join(run_dir, "run_meta.json"), run_meta)

    print(f"\n=== DONE. Summary saved to: {os.path.abspath(summary_path)} ===")
    print("Tip: In your report, cite this CSV + run_meta.json for reproducibility.")


if __name__ == "__main__":
    main()
