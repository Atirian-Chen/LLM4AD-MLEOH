from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


# ============================================================
# basic helpers
# ============================================================
def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def root_dir() -> str:
    # expected to live under MLEOH_3llm/
    return os.path.dirname(os.path.abspath(__file__))


def py() -> str:
    return sys.executable


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def write_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def tee_run(cmd: List[str], cwd: str, log_path: str) -> int:
    ensure_dir(os.path.dirname(log_path))

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.write(f"[CWD] {cwd}\n\n")
        f.flush()

        print(f"\n[RUN] {' '.join(cmd)}", flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

        proc.wait()
        f.write(f"\n[RET] {proc.returncode}\n")
        f.flush()

        print(f"[RET] {proc.returncode}", flush=True)
        return int(proc.returncode)


def common_eoh_args(
    *,
    seeds: List[int],
    timeout: int,
    pop_size: int,
    max_generations: int,
    max_sample_nums: int,
    selection_num: int,
    num_samplers: int,
    num_evaluators: int,
    debug_mode: bool,
    instance_seed: Optional[int],
) -> List[str]:
    args: List[str] = []
    args += ["--seeds"] + [str(s) for s in seeds]
    args += ["--timeout", str(timeout)]
    args += ["--pop_size", str(pop_size)]
    args += ["--max_generations", str(max_generations)]
    args += ["--max_sample_nums", str(max_sample_nums)]
    args += ["--selection_num", str(selection_num)]
    args += ["--num_samplers", str(num_samplers)]
    args += ["--num_evaluators", str(num_evaluators)]
    if instance_seed is not None:
        args += ["--instance_seed", str(instance_seed)]
    if debug_mode:
        args += ["--debug_mode"]
    return args


def cli_from_kv(kv: Dict[str, object]) -> List[str]:
    """
    {"profile_mode": "anonymous", "verify_update": True, "x": 1}
      -> ["--profile_mode","anonymous","--verify_update","--x","1"]
    False / None are omitted.
    """
    out: List[str] = []
    for k, v in kv.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                out.append(flag)
            continue
        if v is None:
            continue
        out += [flag, str(v)]
    return out


# ============================================================
# variants
# ============================================================
@dataclass
class Variant:
    name: str
    description: str
    train_overrides: Dict[str, object]


def build_variants(selected: List[str]) -> List[Variant]:
    all_variants: Dict[str, Variant] = {
        "full": Variant(
            name="full",
            description="Full TextGrad Router Delta baseline",
            train_overrides={},
        ),
        "no_update": Variant(
            name="no_update",
            description="Disable TextGrad policy-text updates via max_updates=0",
            train_overrides={
                "max_updates": 0,
            },
        ),
        "named_profile": Variant(
            name="named_profile",
            description="Use named model profiles instead of anonymous profiles",
            train_overrides={
                "profile_mode": "named",
            },
        ),
        "no_explore": Variant(
            name="no_explore",
            description="Disable explicit exploration and exploration reward",
            train_overrides={
                "explore_epsilon": 0.0,
                "reward_gamma": 0.0,
            },
        ),
    }

    variants: List[Variant] = []
    for name in selected:
        if name not in all_variants:
            raise ValueError(
                f"Unknown variant: {name}. "
                f"Available: {', '.join(all_variants.keys())}"
            )
        variants.append(all_variants[name])
    return variants


# ============================================================
# main run logic
# ============================================================
def build_base_textgrad_train_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "profile_mode": args.profile_mode,
        "update_every": args.update_every,
        "batch_size": args.batch_size,
        "max_updates": args.max_updates,
        "verify_update": args.verify_update,

        "router_temperature": args.router_temperature,
        "critic_temperature": args.critic_temperature,

        "reward_alpha": args.reward_alpha,
        "reward_beta": args.reward_beta,
        "reward_gamma": args.reward_gamma,
        "fail_reward": args.fail_reward,
        "catastrophic_threshold": args.catastrophic_threshold,
        "catastrophic_penalty": args.catastrophic_penalty,
        "reward_shift": args.reward_shift,

        "explore_epsilon": args.explore_epsilon,
        "explore_ucb_c": args.explore_ucb_c,
        "explore_fail_weight": args.explore_fail_weight,

        "min_policy_chars": args.min_policy_chars,
        "max_policy_chars": args.max_policy_chars,
    }


def build_test_runtime_args(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "router_temperature": args.router_temperature,
        "critic_temperature": args.critic_temperature,
    }


def run_variant(
    args: argparse.Namespace,
    run_root: str,
    logs_dir: str,
    variant: Variant,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    variant_root = ensure_dir(os.path.join(run_root, variant.name))
    train_script = os.path.join(root_dir(), "textgrad_router_delta", "train_eoh_obp_textgrad_router3.py")
    test_script = os.path.join(root_dir(), "textgrad_router_delta", "test_eoh_obp_textgrad_router3.py")

    train_exp_name = "train"
    test_exp_name = "test"

    base_train_cfg = build_base_textgrad_train_args(args)
    merged_train_cfg = dict(base_train_cfg)
    merged_train_cfg.update(variant.train_overrides)

    # --------------------------
    # train
    # --------------------------
    train_cmd = [py(), train_script]
    train_cmd += common_eoh_args(
        seeds=args.seeds,
        timeout=args.timeout,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        max_sample_nums=args.learn_train_max_sample_nums,
        selection_num=args.selection_num,
        num_samplers=args.train_num_samplers,
        num_evaluators=args.train_num_evaluators,
        debug_mode=args.debug_mode,
        instance_seed=args.train_instance_seed,
    )
    train_cmd += cli_from_kv(merged_train_cfg)
    train_cmd += ["--out_dir", variant_root, "--exp_name", train_exp_name]

    train_log = os.path.join(logs_dir, f"{variant.name}_train.log")
    print("\n" + "=" * 88)
    print(f"[VARIANT] {variant.name} / train")
    print(f"[DESC]    {variant.description}")
    t0 = time.time()
    train_rc = tee_run(train_cmd, cwd=root_dir(), log_path=train_log)
    train_elapsed = time.time() - t0

    train_dir = os.path.join(variant_root, train_exp_name)
    train_summary = os.path.join(train_dir, "summary.csv")
    state_path = os.path.join(train_dir, "textgrad_router_state.json")
    train_ok = (train_rc == 0) and os.path.isfile(train_summary) and os.path.isfile(state_path)

    rows.append({
        "variant": variant.name,
        "phase": "train",
        "ok": train_ok,
        "return_code": train_rc,
        "elapsed_sec": f"{train_elapsed:.2f}",
        "log_path": os.path.abspath(train_log),
        "out_dir": os.path.abspath(train_dir),
        "summary_csv": os.path.abspath(train_summary) if os.path.isfile(train_summary) else "",
        "state_path": os.path.abspath(state_path) if os.path.isfile(state_path) else "",
        "description": variant.description,
    })

    if not train_ok:
        print(f"[FAIL] {variant.name} / train failed in {fmt_seconds(train_elapsed)}")
        return rows

    print(f"[OK]   {variant.name} / train finished in {fmt_seconds(train_elapsed)}")

    # --------------------------
    # test
    # --------------------------
    test_cmd = [py(), test_script]
    test_cmd += common_eoh_args(
        seeds=args.seeds,
        timeout=args.timeout,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        max_sample_nums=args.main_test_max_sample_nums,
        selection_num=args.selection_num,
        num_samplers=args.test_num_samplers,
        num_evaluators=args.test_num_evaluators,
        debug_mode=args.debug_mode,
        instance_seed=args.test_instance_seed,
    )
    test_cmd += cli_from_kv(build_test_runtime_args(args))
    test_cmd += ["--router_state", state_path]
    test_cmd += ["--out_dir", variant_root, "--exp_name", test_exp_name]

    test_log = os.path.join(logs_dir, f"{variant.name}_test.log")
    print("\n" + "=" * 88)
    print(f"[VARIANT] {variant.name} / test")
    print(f"[DESC]    {variant.description}")
    t1 = time.time()
    test_rc = tee_run(test_cmd, cwd=root_dir(), log_path=test_log)
    test_elapsed = time.time() - t1

    test_dir = os.path.join(variant_root, test_exp_name)
    test_summary = os.path.join(test_dir, "summary.csv")
    test_ok = (test_rc == 0) and os.path.isfile(test_summary)

    rows.append({
        "variant": variant.name,
        "phase": "test",
        "ok": test_ok,
        "return_code": test_rc,
        "elapsed_sec": f"{test_elapsed:.2f}",
        "log_path": os.path.abspath(test_log),
        "out_dir": os.path.abspath(test_dir),
        "summary_csv": os.path.abspath(test_summary) if os.path.isfile(test_summary) else "",
        "state_path": os.path.abspath(state_path),
        "description": variant.description,
    })

    if test_ok:
        print(f"[OK]   {variant.name} / test finished in {fmt_seconds(test_elapsed)}")
    else:
        print(f"[FAIL] {variant.name} / test failed in {fmt_seconds(test_elapsed)}")

    return rows


# ============================================================
# args
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run TextGrad Router Delta ablations (wrapper-only variants)."
    )

    p.add_argument(
        "--variants",
        nargs="+",
        default=["full", "no_update", "named_profile", "no_explore"],
        help="Subset to run. Choices: full no_update named_profile no_explore",
    )
    p.add_argument("--out_base", type=str, default="ablation_results")

    # seeds
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))

    # dataset split seeds
    # 如果你的主实验不是 2024/2025，这里改成和主实验完全一致
    p.add_argument("--train_instance_seed", type=int, default=2024)
    p.add_argument("--test_instance_seed", type=int, default=2025)

    # common EoH settings
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--pop_size", type=int, default=10)
    p.add_argument("--max_generations", type=int, default=6)
    p.add_argument("--selection_num", type=int, default=3)

    # budgets
    p.add_argument("--learn_train_max_sample_nums", type=int, default=300)
    p.add_argument("--main_test_max_sample_nums", type=int, default=1000)

    # parallelism
    p.add_argument("--train_num_samplers", type=int, default=2)
    p.add_argument("--train_num_evaluators", type=int, default=2)
    p.add_argument("--test_num_samplers", type=int, default=4)
    p.add_argument("--test_num_evaluators", type=int, default=4)

    # baseline textgrad-delta config
    p.add_argument("--profile_mode", type=str, default="anonymous", choices=["named", "anonymous"])
    p.add_argument("--update_every", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--max_updates", type=int, default=30)
    p.add_argument("--verify_update", action="store_true", default=False)

    p.add_argument("--router_temperature", type=float, default=0.0)
    p.add_argument("--critic_temperature", type=float, default=0.2)

    p.add_argument("--reward_alpha", type=float, default=0.3)
    p.add_argument("--reward_beta", type=float, default=6.0)
    p.add_argument("--reward_gamma", type=float, default=0.5)
    p.add_argument("--fail_reward", type=float, default=-3.0)
    p.add_argument("--catastrophic_threshold", type=float, default=-4800.0)
    p.add_argument("--catastrophic_penalty", type=float, default=1.0)
    p.add_argument("--reward_shift", type=float, default=0.0)

    p.add_argument("--explore_epsilon", type=float, default=0.25)
    p.add_argument("--explore_ucb_c", type=float, default=1.5)
    p.add_argument("--explore_fail_weight", type=float, default=150.0)

    p.add_argument("--min_policy_chars", type=int, default=200)
    p.add_argument("--max_policy_chars", type=int, default=2600)

    p.add_argument("--debug_mode", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    variants = build_variants(args.variants)

    stamp = now_stamp()
    run_root = ensure_dir(os.path.join(root_dir(), args.out_base, f"ablation_{stamp}"))
    logs_dir = ensure_dir(os.path.join(run_root, "logs"))

    write_json(
        os.path.join(run_root, "run_config.json"),
        {
            "timestamp": stamp,
            "root_dir": root_dir(),
            "args": vars(args),
            "variants": [v.__dict__ for v in variants],
        },
    )

    all_rows: List[Dict[str, object]] = []
    total = len(variants)

    print("=" * 88)
    print("TextGrad Router Delta Ablation Runner")
    print("=" * 88)
    print(f"Run root : {run_root}")
    print(f"Variants : {', '.join(v.name for v in variants)}")
    print(f"Seeds    : {args.seeds}")
    print()

    for i, variant in enumerate(variants, start=1):
        print("\n" + "#" * 88)
        print(f"[{i}/{total}] variant = {variant.name}")
        print("#" * 88)
        rows = run_variant(args, run_root, logs_dir, variant)
        all_rows.extend(rows)

        write_csv(
            os.path.join(run_root, "ablation_index.csv"),
            fieldnames=[
                "variant",
                "phase",
                "ok",
                "return_code",
                "elapsed_sec",
                "log_path",
                "out_dir",
                "summary_csv",
                "state_path",
                "description",
            ],
            rows=all_rows,
        )

    ok_train = sum(1 for r in all_rows if r["phase"] == "train" and r["ok"])
    ok_test = sum(1 for r in all_rows if r["phase"] == "test" and r["ok"])

    print("\n" + "=" * 88)
    print("[DONE] ablation run finished")
    print(f"Run root         : {run_root}")
    print(f"Index CSV        : {os.path.join(run_root, 'ablation_index.csv')}")
    print(f"Train success    : {ok_train}/{len(variants)}")
    print(f"Test success     : {ok_test}/{len(variants)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
#python .\run_ablation.py