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
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

# 你的原始依赖（保持不变）
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


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # optional
        np.random.seed(seed)
    except Exception:
        pass


def _try_set_task_seed(task: Any, seed: int) -> None:
    """
    尽量让 OBP 实例生成/评估可重复。
    由于你本地 llm4ad 版本未知，这里用“多策略尝试”方式兼容。
    """
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

    # 3) 重新构造 task（有的版本 OBPEvaluation(seed=...)）
    #    这一层在外面 run_seed 里做更合适


def _make_task_with_seed(seed: int) -> Any:
    """
    兼容不同 llm4ad 版本的 OBPEvaluation 构造参数。
    """
    try:
        task = OBPEvaluation(seed=seed)
    except TypeError:
        task = OBPEvaluation()
        _try_set_task_seed(task, seed)
    return task


def _safe_get_best_score(method: Any) -> Optional[float]:
    """
    尝试从 EoH 实例里拿 best_score（不同版本字段名可能不一样）。
    """
    for attr in ["best_score", "best_fitness", "best_value", "global_best_score"]:
        v = getattr(method, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    # 有些实现会把 best 放在 profiler 或 history
    return None


def _safe_evaluate(task: Any, program: Any) -> Optional[float]:
    """
    尝试对 best_program 再评估一次得到 best_score。
    注意：不同版本 OBPEvaluation 接口可能不同，所以多策略尝试。
    """
    # program 可能是 str / Program 对象
    prog = program
    if not isinstance(prog, str):
        try:
            prog = str(program)
        except Exception:
            prog = None

    if prog is None:
        return None

    # 常见接口：evaluate / __call__ / run
    for name in ["evaluate", "eval", "run", "__call__"]:
        fn = getattr(task, name, None)
        if callable(fn):
            try:
                out = fn(prog)
                # out 可能是 float 或 dict/tuple
                if isinstance(out, (int, float)):
                    return float(out)
                if isinstance(out, dict):
                    # 常见 key
                    for k in ["score", "best_score", "fitness", "value"]:
                        if k in out and isinstance(out[k], (int, float)):
                            return float(out[k])
                if isinstance(out, (tuple, list)) and out:
                    if isinstance(out[0], (int, float)):
                        return float(out[0])
            except Exception:
                continue
    return None


def _make_llm(args: argparse.Namespace) -> HttpsApi:
    """
    创建 HttpsApi；如果某些参数在你的版本不支持，会自动降级。
    """
    kwargs = dict(
        host=args.host,
        key=args.api_key,
        model=args.model,
        timeout=args.timeout,
    )
    # 某些实现可能支持 temperature / top_p
    extra = {}
    if args.temperature is not None:
        extra["temperature"] = args.temperature
    if args.top_p is not None:
        extra["top_p"] = args.top_p

    try:
        return HttpsApi(**kwargs, **extra)
    except TypeError:
        return HttpsApi(**kwargs)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# -----------------------------
# Main experiment runner
# -----------------------------
def run_one_seed(args: argparse.Namespace, run_dir: str, seed: int) -> Dict[str, Any]:
    seed_dir = os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # 独立的 log_dir（便于你复核每次运行细节）
    log_dir = os.path.join(seed_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    _set_global_seed(seed)

    # 任务与 LLM
    task = _make_task_with_seed(seed)
    llm = _make_llm(args)

    profiler = EoHProfiler(
        log_dir=log_dir,
        log_style="simple",
    )

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

    # 保存 seed 级配置
    seed_cfg = {
        "seed": seed,
        "time_start": _now_iso(),
        "host": args.host,
        "model": args.model,
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
        print(f"[seed={seed}] >>> Start EoH on Online Bin Packing ...")
        best_program = method.run()
        print(f"[seed={seed}] >>> Search finished.")

        # 保存 best program（用于复核和后续复评）
        best_program_text = best_program if isinstance(best_program, str) else str(best_program)
        best_path = os.path.join(seed_dir, "best_program.txt")
        _write_text(best_path, best_program_text)
        sha = _sha256_text(best_program_text)
        _write_text(os.path.join(seed_dir, "best_program.sha256"), sha)

        # 取 best_score：优先从 method 拿，拿不到再评估一次
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
        print(f"[seed={seed}] !!! ERROR:\n{err}")

    result["time_end"] = _now_iso()
    _safe_json_dump(os.path.join(seed_dir, "result.json"), result)
    return result


def write_summary_csv(path: str, rows: list) -> None:
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
    p = argparse.ArgumentParser(description="Reproducible EoH reproduction runner (OBP).")

    # LLM
    # p.add_argument("--host", type=str, default="api.deepseek.com")
    # p.add_argument("--model", type=str, default="deepseek-chat")
    # p.add_argument("--api_key_env", type=str, default="DEEPSEEK_API_KEY",
    #                help="Environment variable name that stores the API key.")
    
    p.add_argument("--host", type=str, default="dashscope.aliyuncs.com")
    p.add_argument("--model", type=str, default="qwen-flash") 
    p.add_argument("--api_key_env", type=str, default="QWEN_API_KEY",
                   help="Environment variable name that stores the API key.")


    p.add_argument("--timeout", type=int, default=60)

    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)

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

    # Repro control
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="List of random seeds to run.")
    p.add_argument("--out_dir", type=str, default="results/eoh_obp_repro")

    # p.add_argument("--exp_name", type=str, default="eoh_obp_single_ds")
    p.add_argument("--exp_name", type=str, default="eoh_obp_single_qw")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"API key not found. Please set environment variable {args.api_key_env}."
        )
    args.api_key = api_key  # attach for _make_llm

    run_id = f"{args.exp_name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 保存 run-level 元信息（便于论文复现说明）
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
        "args": {k: getattr(args, k) for k in vars(args) if k != "api_key"},
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
