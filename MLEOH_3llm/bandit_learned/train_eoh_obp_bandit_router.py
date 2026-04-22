# train_eoh_obp_bandit_router.py
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

from bandit_router import BanditRouterLLM, LinUCB, Decision
from router_trace import RouterTracer


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


class BanditEoH(BaseEoH):
    """
    在 _sample_evaluate_register 里更新 bandit（你的原逻辑不变，只是 arm 变 3）。
    """
    def __init__(self, *args, bandit: LinUCB, fail_penalty: float, reward_mode: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._bandit = bandit
        self._fail_penalty = float(fail_penalty)
        self._reward_mode = str(reward_mode)
        self.best_program_text: Optional[str] = None

    def _reward(self, prev_best: float, score: float) -> float:
        if self._reward_mode == "score":
            return float(score)
        if self._reward_mode == "delta":
            return float(score - prev_best)
        new_best = max(prev_best, score)
        return float(new_best - prev_best)

    def _sample_evaluate_register(self, prompt):
        import time
        from llm4ad.base import TextFunctionProgramConverter
        sample_start = time.time()

        decision: Optional[Decision] = None
        try:
            thought, func = self._sampler.get_thought_and_function(prompt)
            sample_time = time.time() - sample_start

            if hasattr(self._sampler.llm, "pop_last_decision"):
                decision = self._sampler.llm.pop_last_decision()

            if thought is None or func is None:
                if decision is not None:
                    x = np.array(decision.x, dtype=np.float64)
                    self._bandit.update(decision.arm, x, -self._fail_penalty)
                return
        except Exception:
            if decision is not None:
                x = np.array(decision.x, dtype=np.float64)
                self._bandit.update(decision.arm, x, -self._fail_penalty)
            return

        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            if decision is not None:
                x = np.array(decision.x, dtype=np.float64)
                self._bandit.update(decision.arm, x, -self._fail_penalty)
            return

        prev_best = float(self.best_score)
        try:
            score, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time, program
            ).result()
        except Exception:
            if decision is not None:
                x = np.array(decision.x, dtype=np.float64)
                self._bandit.update(decision.arm, x, -self._fail_penalty)
            return

        r = self._reward(prev_best=prev_best, score=float(score))
        if decision is not None:
            x = np.array(decision.x, dtype=np.float64)
            self._bandit.update(decision.arm, x, r)

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

    # backends (3)
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
    p.add_argument("--max_sample_nums", type=int, default=300)
    p.add_argument("--selection_num", type=int, default=3)
    p.add_argument("--use_e2_operator", action="store_true", default=True)
    p.add_argument("--use_m1_operator", action="store_true", default=True)
    p.add_argument("--use_m2_operator", action="store_true", default=True)
    p.add_argument("--num_samplers", type=int, default=2)
    p.add_argument("--num_evaluators", type=int, default=2)

    # bandit
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--reward_mode", type=str, default="score", choices=["improve", "score", "delta"])
    p.add_argument("--fail_penalty", type=float, default=1.0)

    # run
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--exp_name", type=str, default="eoh_obp_bandit_train_3")
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--debug_mode", action="store_true", default=False)
    p.add_argument("--instance_seed", type=int, default=2026)
    return p.parse_args()


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


def run_one_seed(args: argparse.Namespace, run_dir: str, seed: int, router: BanditRouterLLM) -> Dict[str, Any]:
    seed_dir = os.path.join(run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # 每个 seed 单独 trace
    trace_path = os.path.join(seed_dir, "router_trace.csv")
    tracer = RouterTracer(trace_path)
    router.tracer = tracer
    router.set_seed_id(seed)

    _set_global_seed(seed)

    try:
        # task = OBPEvaluation(seed=seed)
        task = OBPEvaluation(seed=seed, instance_seed=args.instance_seed)
    except TypeError:
        task = OBPEvaluation()

    profiler = EoHProfiler(log_dir=seed_dir, log_style="simple")

    cfg = {
        "time_start": _now_iso(),
        "seed": seed,
        "bandit": router.bandit.state_dict(),
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
        method = BanditEoH(
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
            bandit=router.bandit,
            fail_penalty=args.fail_penalty,
            reward_mode=args.reward_mode,
        )

        best_program = method.run()
        best_program_text = best_program if isinstance(best_program, str) else str(best_program)
        best_path = os.path.join(seed_dir, "best_program.txt")
        _write_text(best_path, best_program_text)

        sha = _sha256_text(best_program_text)
        _write_text(os.path.join(seed_dir, "best_program.sha256"), sha)

        best_score = float(getattr(method, "best_score", None))
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

    result["time_end"] = _now_iso()
    _safe_json_dump(os.path.join(seed_dir, "result.json"), result)
    return result


def main():
    args = parse_args()
    exp_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    deepseek = HttpsApi(
        host=args.deepseek_host,
        key=_get_env_or_fail(args.deepseek_key_env),
        model=args.deepseek_model,
        timeout=args.timeout,
    )
    qwen = HttpsApi(
        host=args.qwen_host,
        key=_get_env_or_fail(args.qwen_key_env),
        model=args.qwen_model,
        timeout=args.timeout,
    )
    doubao = HttpsApi(
        host=args.doubao_host,
        key=_get_env_or_fail(args.doubao_key_env),
        model=args.doubao_model,
        timeout=args.timeout,
    )

    from bandit_router import featurize
    x0, _ = featurize("")
    bandit = LinUCB(dim=int(len(x0)), alpha=args.alpha, l2=args.l2, eps=args.eps, n_arms=3)

    router = BanditRouterLLM(
        deepseek_llm=deepseek,
        qwen_llm=qwen,
        doubao_llm=doubao,
        bandit=bandit,
        verbose=False,
        tracer=None,
        seed_id=-1,
    )

    rows = []
    for seed in args.seeds:
        rows.append(run_one_seed(args=args, run_dir=exp_dir, seed=seed, router=router))

    router.save_bandit(os.path.join(exp_dir, "bandit_state.json"))
    write_summary_csv(os.path.join(exp_dir, "summary.csv"), rows)

    router.close()
    print(f"[DONE] train finished. saved to: {exp_dir}")


if __name__ == "__main__":
    main()
#python train_eoh_obp_bandit_router.py --seeds 0 1 2 3 4 --exp_name eoh_obp_bandit_train_3