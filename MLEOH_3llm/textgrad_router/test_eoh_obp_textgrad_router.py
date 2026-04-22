from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import traceback
from typing import Any, Dict, Optional

import numpy as np

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh.eoh import EoH as BaseEoH
from llm4ad.method.eoh import EoHProfiler

from textgrad_router import TextGradRouterLLM, TGTracer


def _now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_json_dump(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_text(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _get_env_or_fail(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise RuntimeError(f"API key not found. Please set env var {name}.")
    return v


class TextGradTestEoH(BaseEoH):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_program_text: Optional[str] = None

    def _sample_evaluate_register(self, prompt):
        import time
        from llm4ad.base import TextFunctionProgramConverter

        sample_start = time.time()
        thought, func = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return

        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return

        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time, program
        ).result()

        if float(score) >= float(self.best_score):
            self.best_program_text = str(program)

        func.score = score
        self.best_score = max(self.best_score, score)
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time

        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)
            self._tot_sample_nums += 1

        self._population.register_function(func)

    def run(self):
        super().run()
        return self.best_program_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # router LLM (decision) - must match train side for fairness
    p.add_argument("--router_host", type=str, default="ark.cn-beijing.volces.com")
    p.add_argument("--router_model", type=str, default="doubao-seed-2-0-mini-260215")
    p.add_argument("--router_key_env", type=str, default="ARK_API_KEY")

    # critic is unused in test but keep for constructor compatibility
    p.add_argument("--critic_host", type=str, default="api.deepseek.com")
    p.add_argument("--critic_model", type=str, default="deepseek-chat")
    p.add_argument("--critic_key_env", type=str, default="DEEPSEEK_API_KEY")

    # backends
    p.add_argument("--deepseek_host", type=str, default="api.deepseek.com")
    p.add_argument("--deepseek_model", type=str, default="deepseek-chat")
    p.add_argument("--deepseek_key_env", type=str, default="DEEPSEEK_API_KEY")

    p.add_argument("--qwen_host", type=str, default="dashscope.aliyuncs.com")
    p.add_argument("--qwen_model", type=str, default="qwen-flash")
    p.add_argument("--qwen_key_env", type=str, default="QWEN_API_KEY")

    p.add_argument("--doubao_host", type=str, default="ark.cn-beijing.volces.com")
    p.add_argument("--doubao_model", type=str, default="doubao-seed-2-0-mini-260215")
    p.add_argument("--doubao_key_env", type=str, default="ARK_API_KEY")

    p.add_argument("--timeout", type=int, default=60)

    # eoh
    p.add_argument("--pop_size", type=int, default=10)
    p.add_argument("--max_generations", type=int, default=6)
    p.add_argument("--max_sample_nums", type=int, default=1000)
    p.add_argument("--selection_num", type=int, default=3)
    p.add_argument("--use_e2_operator", action="store_true", default=True)
    p.add_argument("--use_m1_operator", action="store_true", default=True)
    p.add_argument("--use_m2_operator", action="store_true", default=True)
    p.add_argument("--num_samplers", type=int, default=4)
    p.add_argument("--num_evaluators", type=int, default=4)

    p.add_argument("--router_state", type=str, required=True)

    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--exp_name", type=str, default="eoh_obp_textgrad_test_3")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--debug_mode", action="store_true", default=False)
    return p.parse_args()


def write_summary_csv(path: str, rows: list) -> None:
    fieldnames = ["seed", "ok", "best_score", "best_program_sha256", "best_program_path", "time_start", "time_end", "error"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def run_one_seed(args: argparse.Namespace, exp_dir: str, seed: int, router: TextGradRouterLLM) -> Dict[str, Any]:
    seed_dir = os.path.join(exp_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    trace_path = os.path.join(seed_dir, "router_trace.csv")
    router.tracer = TGTracer(trace_path)

    _set_global_seed(seed)
    try:
        task = OBPEvaluation(seed=seed)
    except TypeError:
        task = OBPEvaluation()

    profiler = EoHProfiler(log_dir=seed_dir, log_style="simple")

    cfg = {
        "time_start": _now_iso(),
        "seed": seed,
        "router_state": os.path.abspath(args.router_state),
        "router_trace_path": os.path.abspath(trace_path),
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
        },
    }
    _safe_json_dump(os.path.join(seed_dir, "config.json"), cfg)

    result: Dict[str, Any] = {
        "seed": seed,
        "ok": False,
        "best_score": None,
        "best_program_sha256": None,
        "best_program_path": None,
        "error": None,
        "time_start": cfg["time_start"],
        "time_end": None,
    }

    try:
        method = TextGradTestEoH(
            llm=router,
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

        best_program = method.run()
        best_text = best_program if isinstance(best_program, str) else str(best_program)
        best_path = os.path.join(seed_dir, "best_program.txt")
        _write_text(best_path, best_text)
        sha = _sha256_text(best_text)
        _write_text(os.path.join(seed_dir, "best_program.sha256"), sha)

        best_score = float(getattr(method, "best_score", None))
        result.update(ok=True, best_score=best_score, best_program_sha256=sha, best_program_path=os.path.abspath(best_path))

    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        result["error"] = err
        _write_text(os.path.join(seed_dir, "error.txt"), err)

    result["time_end"] = _now_iso()
    _safe_json_dump(os.path.join(seed_dir, "result.json"), result)
    return result


def main():
    args = parse_args()
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    state = TextGradRouterLLM.load_state(args.router_state)
    policy_text = state.get("policy_text", "")
    update_every = int(state.get("update_every", 25))
    batch_size = int(state.get("batch_size", 25))
    max_updates = int(state.get("max_updates", 0))
    reward_shift = float(state.get("reward_shift", 0.0))

    router_llm = HttpsApi(host=args.router_host, key=_get_env_or_fail(args.router_key_env), model=args.router_model, timeout=args.timeout, temperature=0.0)
    critic_llm = HttpsApi(host=args.critic_host, key=_get_env_or_fail(args.critic_key_env), model=args.critic_model, timeout=args.timeout, temperature=0.2)

    deepseek = HttpsApi(host=args.deepseek_host, key=_get_env_or_fail(args.deepseek_key_env), model=args.deepseek_model, timeout=args.timeout)
    qwen = HttpsApi(host=args.qwen_host, key=_get_env_or_fail(args.qwen_key_env), model=args.qwen_model, timeout=args.timeout)
    doubao = HttpsApi(host=args.doubao_host, key=_get_env_or_fail(args.doubao_key_env), model=args.doubao_model, timeout=args.timeout)

    router = TextGradRouterLLM(
        router_llm=router_llm,
        critic_llm=critic_llm,
        deepseek_llm=deepseek,
        qwen_llm=qwen,
        doubao_llm=doubao,
        policy_text=policy_text,
        tracer=None,
        verbose=False,
        freeze=True,              # 测试冻结
        update_every=update_every,
        batch_size=batch_size,
        max_updates=max_updates,
        reward_shift=reward_shift,
    )
    router.policy_version = int(state.get("policy_version", 0))

    rows = []
    for seed in args.seeds:
        rows.append(run_one_seed(args=args, exp_dir=exp_dir, seed=seed, router=router))

    write_summary_csv(os.path.join(exp_dir, "summary.csv"), rows)
    router.close()
    print(f"[DONE] test finished. saved to: {exp_dir}")


if __name__ == "__main__":
    main()
    #python ./test_eoh_obp_textgrad_router.py --router_state runs/eoh_obp_textgrad_train_3/textgrad_router_state.json --seeds 0 1 2 3 4 --exp_name eoh_obp_textgrad_test_3 --max_sample_nums 1000