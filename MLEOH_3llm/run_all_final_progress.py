from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ============================================================
# Helpers
# ============================================================
def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def abspath(path: str) -> str:
    return os.path.abspath(path)


def root_dir() -> str:
    # This file is expected to live under MLEOH_3llm/
    return os.path.dirname(os.path.abspath(__file__))


def py() -> str:
    return sys.executable


def list_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    out: List[str] = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            out.append(full)
    return out


def newest_dir_with_prefix(parent: str, prefix: str) -> Optional[str]:
    cands: List[Tuple[float, str]] = []
    for d in list_dirs(parent):
        base = os.path.basename(d)
        if base.startswith(prefix + "_"):
            try:
                mtime = os.path.getmtime(d)
            except Exception:
                mtime = 0.0
            cands.append((mtime, d))
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]


def tee_run(cmd: List[str], cwd: str, log_path: str) -> int:
    ensure_dir(os.path.dirname(log_path))

    # Force child python processes to flush stdout/stderr promptly
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w", encoding="utf-8", newline="") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.write(f"[CWD] {cwd}\n\n")
        f.flush()

        # Optional: print command before running, so you know exactly what started
        print(f"[RUN] {' '.join(cmd)}", flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()   # <- make terminal output immediate
            f.write(line)
            f.flush()            # <- make log file update immediately too

        proc.wait()
        f.write(f"\n[RET] {proc.returncode}\n")
        f.flush()

        print(f"[RET] {proc.returncode}", flush=True)
        return int(proc.returncode)


def write_text(path: str, s: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def append_text(path: str, s: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(s)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def add_instance_seed_arg(cmd: List[str], instance_seed: Optional[int]) -> None:
    # Assumes each runner script has already been modified to accept --instance_seed.
    if instance_seed is not None:
        cmd += ["--instance_seed", str(instance_seed)]


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
    if debug_mode:
        args += ["--debug_mode"]
    return args


def fmt_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def progress_bar(done: int, total: int, width: int = 28) -> str:
    total = max(1, total)
    done = max(0, min(done, total))
    filled = int(round(width * done / total))
    return "#" * filled + "-" * (width - filled)


def print_step_banner(index: int, total: int, method: str, phase: str) -> float:
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = progress_bar(index - 1, total)
    print("\n" + "=" * 88)
    print(f"[{ts}] STEP {index}/{total}  [{bar}]  {method} / {phase}")
    print("=" * 88)
    return time.time()


def print_step_footer(index: int, total: int, sr: 'StepResult', started_at: float, ok_count: int) -> None:
    elapsed = time.time() - started_at
    bar = progress_bar(index, total)
    status = "OK" if sr.ok else "FAIL"
    print(f"\n[{status}] finished STEP {index}/{total} in {fmt_seconds(elapsed)} :: {sr.method} / {sr.phase}")
    print(f"Progress: [{bar}] {index}/{total} complete | successes: {ok_count}/{index}")
    if sr.summary_csv:
        print(f"Summary:  {abspath(sr.summary_csv)}")
    print()


@dataclass
class StepResult:
    method: str
    phase: str
    ok: bool
    cmd: List[str]
    log_path: str
    out_dir: Optional[str]
    summary_csv: Optional[str]
    state_path: Optional[str]
    note: str = ""


# ============================================================
# Runners
# ============================================================
def run_replication(args: argparse.Namespace, run_root: str, logs_dir: str) -> StepResult:
    method = "replication"
    phase = "run"
    out_dir = ensure_dir(os.path.join(run_root, method))

    script = os.path.join(root_dir(), "replication", "run_eoh_obp_repro.py")
    cmd = [py(), script]
    cmd += common_eoh_args(
        seeds=args.seeds,
        timeout=args.timeout,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        max_sample_nums=args.main_max_sample_nums,
        selection_num=args.selection_num,
        num_samplers=args.main_num_samplers,
        num_evaluators=args.main_num_evaluators,
        debug_mode=args.debug_mode,
    )
    add_instance_seed_arg(cmd, args.test_instance_seed)
    cmd += ["--out_dir", out_dir, "--exp_name", args.rep_exp_name]

    log_path = os.path.join(logs_dir, f"{method}.log")
    rc = tee_run(cmd, cwd=root_dir(), log_path=log_path)

    run_dir = newest_dir_with_prefix(out_dir, args.rep_exp_name)
    summary_csv = os.path.join(run_dir, "summary_all.csv") if run_dir else None
    ok = (rc == 0) and (run_dir is not None) and (summary_csv is not None) and os.path.isfile(summary_csv)

    return StepResult(
        method=method,
        phase=phase,
        ok=ok,
        cmd=cmd,
        log_path=log_path,
        out_dir=run_dir,
        summary_csv=summary_csv,
        state_path=None,
        note="Main-table single-model baseline. Uses test_instance_seed and unified main budget.",
    )


def run_random(args: argparse.Namespace, run_root: str, logs_dir: str) -> StepResult:
    method = "random"
    phase = "run"
    out_dir = ensure_dir(os.path.join(run_root, method))

    script = os.path.join(root_dir(), "random", "run_eoh_obp_random_router.py")
    cmd = [py(), script]
    cmd += common_eoh_args(
        seeds=args.seeds,
        timeout=args.timeout,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        max_sample_nums=args.main_max_sample_nums,
        selection_num=args.selection_num,
        num_samplers=args.main_num_samplers,
        num_evaluators=args.main_num_evaluators,
        debug_mode=args.debug_mode,
    )
    add_instance_seed_arg(cmd, args.test_instance_seed)
    cmd += [
        "--p_deepseek", str(args.random_p_deepseek),
        "--p_qwen", str(args.random_p_qwen),
    ]
    if args.random_p_doubao is not None:
        cmd += ["--p_doubao", str(args.random_p_doubao)]
    cmd += ["--out_dir", out_dir, "--exp_name", args.random_exp_name]

    log_path = os.path.join(logs_dir, f"{method}.log")
    rc = tee_run(cmd, cwd=root_dir(), log_path=log_path)

    run_dir = newest_dir_with_prefix(out_dir, args.random_exp_name)
    summary_csv = os.path.join(run_dir, "summary.csv") if run_dir else None
    ok = (rc == 0) and (run_dir is not None) and (summary_csv is not None) and os.path.isfile(summary_csv)

    return StepResult(method, phase, ok, cmd, log_path, run_dir, summary_csv, None)


def run_static_rule(args: argparse.Namespace, run_root: str, logs_dir: str) -> StepResult:
    method = "static_rule"
    phase = "run"
    out_dir = ensure_dir(os.path.join(run_root, method))

    script = os.path.join(root_dir(), "static_rule", "run_eoh_obp_rule_router.py")
    cmd = [py(), script]
    cmd += common_eoh_args(
        seeds=args.seeds,
        timeout=args.timeout,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        max_sample_nums=args.main_max_sample_nums,
        selection_num=args.selection_num,
        num_samplers=args.main_num_samplers,
        num_evaluators=args.main_num_evaluators,
        debug_mode=args.debug_mode,
    )
    add_instance_seed_arg(cmd, args.test_instance_seed)
    cmd += [
        "--long_prompt_chars", str(args.rule_long_prompt_chars),
        "--long_prompt_lines", str(args.rule_long_prompt_lines),
        "--dense_code_markers", str(args.rule_dense_code_markers),
        "--default_backend", str(args.rule_default_backend),
    ]
    if args.rule_verbose:
        cmd += ["--router_verbose"]
    cmd += ["--out_dir", out_dir, "--exp_name", args.rule_exp_name]

    log_path = os.path.join(logs_dir, f"{method}.log")
    rc = tee_run(cmd, cwd=root_dir(), log_path=log_path)

    run_dir = newest_dir_with_prefix(out_dir, args.rule_exp_name)
    summary_csv = os.path.join(run_dir, "summary.csv") if run_dir else None
    ok = (rc == 0) and (run_dir is not None) and (summary_csv is not None) and os.path.isfile(summary_csv)

    return StepResult(method, phase, ok, cmd, log_path, run_dir, summary_csv, None)


def run_bandit(args: argparse.Namespace, run_root: str, logs_dir: str) -> Tuple[StepResult, StepResult]:
    method = "bandit_learned"
    out_dir = ensure_dir(os.path.join(run_root, method))

    # train
    train_script = os.path.join(root_dir(), "bandit_learned", "train_eoh_obp_bandit_router.py")
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
    )
    add_instance_seed_arg(train_cmd, args.train_instance_seed)
    train_cmd += ["--reward_mode", str(args.bandit_reward_mode)]
    train_cmd += ["--alpha", str(args.bandit_alpha)]
    train_cmd += ["--l2", str(args.bandit_l2)]
    train_cmd += ["--eps", str(args.bandit_eps)]
    train_cmd += ["--fail_penalty", str(args.bandit_fail_penalty)]
    train_cmd += ["--out_dir", out_dir, "--exp_name", args.bandit_train_exp_name]

    train_log = os.path.join(logs_dir, f"{method}_train.log")
    train_rc = tee_run(train_cmd, cwd=root_dir(), log_path=train_log)

    train_dir = os.path.join(out_dir, args.bandit_train_exp_name)
    bandit_state = os.path.join(train_dir, "bandit_state.json")
    train_summary = os.path.join(train_dir, "summary.csv")
    train_ok = (train_rc == 0) and os.path.isfile(bandit_state)
    train_res = StepResult(method, "train", train_ok, train_cmd, train_log, train_dir, train_summary, bandit_state)

    # test
    test_script = os.path.join(root_dir(), "bandit_learned", "test_eoh_obp_bandit_router.py")
    test_cmd = [py(), test_script]
    test_cmd += common_eoh_args(
        seeds=args.seeds,
        timeout=args.timeout,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        max_sample_nums=args.main_max_sample_nums,
        selection_num=args.selection_num,
        num_samplers=args.test_num_samplers,
        num_evaluators=args.test_num_evaluators,
        debug_mode=args.debug_mode,
    )
    add_instance_seed_arg(test_cmd, args.test_instance_seed)
    test_cmd += ["--bandit_state", bandit_state]
    test_cmd += ["--out_dir", out_dir, "--exp_name", args.bandit_test_exp_name]

    test_log = os.path.join(logs_dir, f"{method}_test.log")
    test_rc = tee_run(test_cmd, cwd=root_dir(), log_path=test_log)

    test_dir = os.path.join(out_dir, args.bandit_test_exp_name)
    test_summary = os.path.join(test_dir, "summary.csv")
    test_ok = (test_rc == 0) and os.path.isfile(test_summary)
    test_res = StepResult(method, "test", test_ok, test_cmd, test_log, test_dir, test_summary, bandit_state)

    return train_res, test_res


def run_statistic_llmrouter(args: argparse.Namespace, run_root: str, logs_dir: str) -> Tuple[StepResult, StepResult]:
    method = "statistic_llmrouter"
    out_dir = ensure_dir(os.path.join(run_root, method))

    # train
    train_script = os.path.join(root_dir(), "statistic_llmrouter", "train_eoh_obp_llm_router.py")
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
    )
    add_instance_seed_arg(train_cmd, args.train_instance_seed)
    train_cmd += ["--reward_mode", str(args.llmrouter_reward_mode)]
    train_cmd += ["--fail_penalty", str(args.llmrouter_fail_penalty)]
    train_cmd += ["--out_dir", out_dir, "--exp_name", args.llmrouter_train_exp_name]

    train_log = os.path.join(logs_dir, f"{method}_train.log")
    train_rc = tee_run(train_cmd, cwd=root_dir(), log_path=train_log)

    train_dir = os.path.join(out_dir, args.llmrouter_train_exp_name)
    stats_path = os.path.join(train_dir, "router_stats.json")
    train_summary = os.path.join(train_dir, "summary.csv")
    train_ok = (train_rc == 0) and os.path.isfile(stats_path)
    train_res = StepResult(method, "train", train_ok, train_cmd, train_log, train_dir, train_summary, stats_path)

    # test
    test_script = os.path.join(root_dir(), "statistic_llmrouter", "test_eoh_obp_llm_router.py")
    test_cmd = [py(), test_script]
    test_cmd += common_eoh_args(
        seeds=args.seeds,
        timeout=args.timeout,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        max_sample_nums=args.main_max_sample_nums,
        selection_num=args.selection_num,
        num_samplers=args.test_num_samplers,
        num_evaluators=args.test_num_evaluators,
        debug_mode=args.debug_mode,
    )
    add_instance_seed_arg(test_cmd, args.test_instance_seed)
    test_cmd += ["--router_stats", stats_path]
    test_cmd += ["--out_dir", out_dir, "--exp_name", args.llmrouter_test_exp_name]

    test_log = os.path.join(logs_dir, f"{method}_test.log")
    test_rc = tee_run(test_cmd, cwd=root_dir(), log_path=test_log)

    test_dir = os.path.join(out_dir, args.llmrouter_test_exp_name)
    test_summary = os.path.join(test_dir, "summary.csv")
    test_ok = (test_rc == 0) and os.path.isfile(test_summary)
    test_res = StepResult(method, "test", test_ok, test_cmd, test_log, test_dir, test_summary, stats_path)

    return train_res, test_res


def run_textgrad_delta(args: argparse.Namespace, run_root: str, logs_dir: str) -> Tuple[StepResult, StepResult]:
    method = "textgrad_router_delta"
    out_dir = ensure_dir(os.path.join(run_root, method))

    # train
    train_script = os.path.join(root_dir(), "textgrad_router_delta", "train_eoh_obp_textgrad_router3.py")
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
    )
    add_instance_seed_arg(train_cmd, args.train_instance_seed)

    # risk/exploration-oriented v3(delta) settings
    train_cmd += ["--profile_mode", str(args.textgrad_delta_profile_mode)]
    train_cmd += ["--update_every", str(args.textgrad_delta_update_every)]
    train_cmd += ["--batch_size", str(args.textgrad_delta_batch_size)]
    train_cmd += ["--max_updates", str(args.textgrad_delta_max_updates)]
    if args.textgrad_delta_verify_update:
        train_cmd += ["--verify_update"]

    train_cmd += ["--router_temperature", str(args.textgrad_delta_router_temperature)]
    train_cmd += ["--critic_temperature", str(args.textgrad_delta_critic_temperature)]

    train_cmd += ["--reward_alpha", str(args.textgrad_delta_reward_alpha)]
    train_cmd += ["--reward_beta", str(args.textgrad_delta_reward_beta)]
    train_cmd += ["--reward_gamma", str(args.textgrad_delta_reward_gamma)]
    train_cmd += ["--fail_reward", str(args.textgrad_delta_fail_reward)]
    train_cmd += ["--catastrophic_threshold", str(args.textgrad_delta_catastrophic_threshold)]
    train_cmd += ["--catastrophic_penalty", str(args.textgrad_delta_catastrophic_penalty)]
    train_cmd += ["--reward_shift", str(args.textgrad_delta_reward_shift)]

    train_cmd += ["--explore_epsilon", str(args.textgrad_delta_explore_epsilon)]
    train_cmd += ["--explore_ucb_c", str(args.textgrad_delta_explore_ucb_c)]
    train_cmd += ["--explore_fail_weight", str(args.textgrad_delta_explore_fail_weight)]

    train_cmd += ["--min_policy_chars", str(args.textgrad_delta_min_policy_chars)]
    train_cmd += ["--max_policy_chars", str(args.textgrad_delta_max_policy_chars)]
    train_cmd += ["--out_dir", out_dir, "--exp_name", args.textgrad_delta_train_exp_name]

    train_log = os.path.join(logs_dir, f"{method}_train.log")
    train_rc = tee_run(train_cmd, cwd=root_dir(), log_path=train_log)

    train_dir = os.path.join(out_dir, args.textgrad_delta_train_exp_name)
    state_path = os.path.join(train_dir, "textgrad_router_state.json")
    train_summary = os.path.join(train_dir, "summary.csv")
    train_ok = (train_rc == 0) and os.path.isfile(state_path)
    train_res = StepResult(method, "train", train_ok, train_cmd, train_log, train_dir, train_summary, state_path)

    # test
    test_script = os.path.join(root_dir(), "textgrad_router_delta", "test_eoh_obp_textgrad_router3.py")
    test_cmd = [py(), test_script]
    test_cmd += common_eoh_args(
        seeds=args.seeds,
        timeout=args.timeout,
        pop_size=args.pop_size,
        max_generations=args.max_generations,
        max_sample_nums=args.main_max_sample_nums,
        selection_num=args.selection_num,
        num_samplers=args.test_num_samplers,
        num_evaluators=args.test_num_evaluators,
        debug_mode=args.debug_mode,
    )
    add_instance_seed_arg(test_cmd, args.test_instance_seed)
    test_cmd += ["--router_temperature", str(args.textgrad_delta_router_temperature)]
    test_cmd += ["--critic_temperature", str(args.textgrad_delta_critic_temperature)]
    test_cmd += ["--router_state", state_path]
    test_cmd += ["--out_dir", out_dir, "--exp_name", args.textgrad_delta_test_exp_name]

    test_log = os.path.join(logs_dir, f"{method}_test.log")
    test_rc = tee_run(test_cmd, cwd=root_dir(), log_path=test_log)

    test_dir = os.path.join(out_dir, args.textgrad_delta_test_exp_name)
    test_summary = os.path.join(test_dir, "summary.csv")
    test_ok = (test_rc == 0) and os.path.isfile(test_summary)
    test_res = StepResult(method, "test", test_ok, test_cmd, test_log, test_dir, test_summary, state_path)

    return train_res, test_res


# ============================================================
# Aggregation
# ============================================================
def step_to_index_block(sr: StepResult) -> str:
    lines: List[str] = []
    lines.append(f"== {sr.method} / {sr.phase} ==")
    lines.append(f"ok: {sr.ok}")
    lines.append(f"log: {abspath(sr.log_path)}")
    lines.append(f"cmd: {' '.join(sr.cmd)}")
    if sr.out_dir:
        lines.append(f"out_dir: {abspath(sr.out_dir)}")
    if sr.summary_csv:
        lines.append(f"summary: {abspath(sr.summary_csv)}")
    if sr.state_path:
        lines.append(f"state: {abspath(sr.state_path)}")
    if sr.note:
        lines.append(f"note: {sr.note}")
    lines.append("")
    return "\n".join(lines)


def merge_summaries(run_root: str, steps: List[StepResult]) -> str:
    rows_out: List[Dict[str, str]] = []
    for step in steps:
        if not step.summary_csv or not os.path.isfile(step.summary_csv):
            continue
        rows = read_csv_rows(step.summary_csv)
        for row in rows:
            rr = dict(row)
            rr["method"] = step.method
            rr["phase"] = step.phase
            rows_out.append(rr)

    fieldset = set()
    for row in rows_out:
        fieldset.update(row.keys())

    preferred = [
        "method", "phase", "backend", "seed", "ok", "best_score",
        "best_program_sha256", "best_program_path", "time_start", "time_end", "error"
    ]
    fieldnames: List[str] = [k for k in preferred if k in fieldset]
    for k in sorted(fieldset):
        if k not in fieldnames:
            fieldnames.append(k)

    out_path = os.path.join(run_root, "summary_all_methods.csv")
    write_csv(out_path, fieldnames, rows_out)
    return out_path


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "run_all.py - final-paper runner for OBP experiments "
        "(replication/random/static_rule/bandit/statistic_llmrouter/textgrad_delta)"
    )

    # 10 seeds by default for more stable results
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out_base", type=str, default="all_results")

    # Split between train/test instance distributions
    p.add_argument("--train_instance_seed", type=int, default=2026)
    p.add_argument("--test_instance_seed", type=int, default=2025)

    # Common EoH controls
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--pop_size", type=int, default=10)
    p.add_argument("--max_generations", type=int, default=6)
    p.add_argument("--selection_num", type=int, default=3)

    # Unified main-table budget; learning phase kept separate
    p.add_argument("--main_max_sample_nums", type=int, default=1000)
    p.add_argument("--learn_train_max_sample_nums", type=int, default=300)

    # Parallelism
    p.add_argument("--main_num_samplers", type=int, default=4)
    p.add_argument("--main_num_evaluators", type=int, default=4)
    p.add_argument("--train_num_samplers", type=int, default=2)
    p.add_argument("--train_num_evaluators", type=int, default=2)
    p.add_argument("--test_num_samplers", type=int, default=4)
    p.add_argument("--test_num_evaluators", type=int, default=4)

    # Random router probabilities
    p.add_argument("--random_p_deepseek", type=float, default=1.0 / 3.0)
    p.add_argument("--random_p_qwen", type=float, default=1.0 / 3.0)
    p.add_argument("--random_p_doubao", type=float, default=None)

    # Static-rule router
    p.add_argument("--rule_long_prompt_chars", type=int, default=12000)
    p.add_argument("--rule_long_prompt_lines", type=int, default=260)
    p.add_argument("--rule_dense_code_markers", type=int, default=18)
    p.add_argument("--rule_default_backend", type=str, default="doubao", choices=["deepseek", "qwen", "doubao"])
    p.add_argument("--rule_verbose", action="store_true", default=False)

    # Bandit router
    p.add_argument("--bandit_reward_mode", type=str, default="score", choices=["improve", "score", "delta"])
    p.add_argument("--bandit_fail_penalty", type=float, default=1.0)
    p.add_argument("--bandit_alpha", type=float, default=1.0)
    p.add_argument("--bandit_l2", type=float, default=1.0)
    p.add_argument("--bandit_eps", type=float, default=0.05)

    # Statistic router
    p.add_argument("--llmrouter_reward_mode", type=str, default="score", choices=["score", "improve", "delta"])
    p.add_argument("--llmrouter_fail_penalty", type=float, default=1.0)

    # TextGrad delta router (risk / exploration oriented defaults from your existing risk config)
    p.add_argument("--textgrad_delta_profile_mode", type=str, default="anonymous", choices=["named", "anonymous"])
    p.add_argument("--textgrad_delta_update_every", type=int, default=25)
    p.add_argument("--textgrad_delta_batch_size", type=int, default=25)
    p.add_argument("--textgrad_delta_max_updates", type=int, default=30)
    p.add_argument("--textgrad_delta_verify_update", action="store_true", default=False)

    p.add_argument("--textgrad_delta_router_temperature", type=float, default=0.0)
    p.add_argument("--textgrad_delta_critic_temperature", type=float, default=0.2)

    p.add_argument("--textgrad_delta_reward_alpha", type=float, default=0.3)
    p.add_argument("--textgrad_delta_reward_beta", type=float, default=6.0)
    p.add_argument("--textgrad_delta_reward_gamma", type=float, default=0.5)
    p.add_argument("--textgrad_delta_fail_reward", type=float, default=-3.0)
    p.add_argument("--textgrad_delta_catastrophic_threshold", type=float, default=-4800.0)
    p.add_argument("--textgrad_delta_catastrophic_penalty", type=float, default=1.0)
    p.add_argument("--textgrad_delta_reward_shift", type=float, default=0.0)

    p.add_argument("--textgrad_delta_explore_epsilon", type=float, default=0.25)
    p.add_argument("--textgrad_delta_explore_ucb_c", type=float, default=1.5)
    p.add_argument("--textgrad_delta_explore_fail_weight", type=float, default=150.0)

    p.add_argument("--textgrad_delta_min_policy_chars", type=int, default=200)
    p.add_argument("--textgrad_delta_max_policy_chars", type=int, default=2600)

    # Experiment names
    p.add_argument("--rep_exp_name", type=str, default="eoh_obp_repro_main1000_test2025_s10")
    p.add_argument("--random_exp_name", type=str, default="eoh_obp_random_main1000_test2025_s10")
    p.add_argument("--rule_exp_name", type=str, default="eoh_obp_rule_main1000_test2025_s10")

    p.add_argument("--bandit_train_exp_name", type=str, default="eoh_obp_bandit_train300_train2025_s10")
    p.add_argument("--bandit_test_exp_name", type=str, default="eoh_obp_bandit_test1000_test2026_s10")

    p.add_argument("--llmrouter_train_exp_name", type=str, default="eoh_obp_llmrouter_train300_train2025_s10")
    p.add_argument("--llmrouter_test_exp_name", type=str, default="eoh_obp_llmrouter_test1000_test2026_s10")

    p.add_argument("--textgrad_delta_train_exp_name", type=str, default="eoh_obp_textgrad_delta_train300_risk_train2024_s10")
    p.add_argument("--textgrad_delta_test_exp_name", type=str, default="eoh_obp_textgrad_delta_test1000_risk_test2025_s10")

    # Behavior
    p.add_argument("--stop_on_error", action="store_true", default=False)
    p.add_argument("--debug_mode", action="store_true", default=False)

    return p.parse_args()


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()

    run_id = now_stamp()
    run_root = ensure_dir(os.path.join(root_dir(), args.out_base, run_id))
    logs_dir = ensure_dir(os.path.join(run_root, "logs"))
    index_path = os.path.join(run_root, "RESULT_INDEX.txt")
    overall_started_at = time.time()

    write_text(
        index_path,
        (
            f"RUN_ID: {run_id}\n"
            f"ROOT: {abspath(run_root)}\n"
            f"SEEDS: {args.seeds}\n"
            f"TRAIN_INSTANCE_SEED: {args.train_instance_seed}\n"
            f"TEST_INSTANCE_SEED: {args.test_instance_seed}\n"
            f"MAIN_MAX_SAMPLE_NUMS: {args.main_max_sample_nums}\n"
            f"LEARN_TRAIN_MAX_SAMPLE_NUMS: {args.learn_train_max_sample_nums}\n\n"
        ),
    )

    plan: List[Tuple[str, str, callable]] = [
        ("replication", "run", lambda: run_replication(args, run_root, logs_dir)),
        ("random", "run", lambda: run_random(args, run_root, logs_dir)),
        ("static_rule", "run", lambda: run_static_rule(args, run_root, logs_dir)),
        ("bandit_learned", "train", lambda: run_bandit(args, run_root, logs_dir)[0]),
        ("bandit_learned", "test", lambda: run_bandit(args, run_root, logs_dir)[1]),
        ("statistic_llmrouter", "train", lambda: run_statistic_llmrouter(args, run_root, logs_dir)[0]),
        ("statistic_llmrouter", "test", lambda: run_statistic_llmrouter(args, run_root, logs_dir)[1]),
        ("textgrad_router_delta", "train", lambda: run_textgrad_delta(args, run_root, logs_dir)[0]),
        ("textgrad_router_delta", "test", lambda: run_textgrad_delta(args, run_root, logs_dir)[1]),
    ]

    # Replace paired-function lambdas with cached execution so train/test only runs once.
    cached_pair_results: Dict[str, Tuple[StepResult, StepResult]] = {}

    def get_pair_cached(key: str, fn):
        if key not in cached_pair_results:
            cached_pair_results[key] = fn()
        return cached_pair_results[key]

    plan = [
        ("replication", "run", lambda: run_replication(args, run_root, logs_dir)),
        ("random", "run", lambda: run_random(args, run_root, logs_dir)),
        ("static_rule", "run", lambda: run_static_rule(args, run_root, logs_dir)),
        ("bandit_learned", "train", lambda: get_pair_cached("bandit_learned", lambda: run_bandit(args, run_root, logs_dir))[0]),
        ("bandit_learned", "test", lambda: get_pair_cached("bandit_learned", lambda: run_bandit(args, run_root, logs_dir))[1]),
        ("statistic_llmrouter", "train", lambda: get_pair_cached("statistic_llmrouter", lambda: run_statistic_llmrouter(args, run_root, logs_dir))[0]),
        ("statistic_llmrouter", "test", lambda: get_pair_cached("statistic_llmrouter", lambda: run_statistic_llmrouter(args, run_root, logs_dir))[1]),
        ("textgrad_router_delta", "train", lambda: get_pair_cached("textgrad_router_delta", lambda: run_textgrad_delta(args, run_root, logs_dir))[0]),
        ("textgrad_router_delta", "test", lambda: get_pair_cached("textgrad_router_delta", lambda: run_textgrad_delta(args, run_root, logs_dir))[1]),
    ]

    total_steps = len(plan)
    steps: List[StepResult] = []
    ok_count = 0

    print("=" * 88)
    print("OBP final-paper run_all started")
    print(f"Run root: {abspath(run_root)}")
    print(f"Seeds: {args.seeds}")
    print(f"Train/Test instance seeds: {args.train_instance_seed} / {args.test_instance_seed}")
    print(f"Unified main budget: {args.main_max_sample_nums} | learning-train budget: {args.learn_train_max_sample_nums}")
    print(f"Total step count: {total_steps}")
    print("=" * 88)

    def record(step: StepResult) -> None:
        nonlocal ok_count
        steps.append(step)
        append_text(index_path, step_to_index_block(step))
        if step.ok:
            ok_count += 1
        if (not step.ok) and args.stop_on_error:
            append_text(index_path, "[STOP] stop_on_error=True, aborting.\n")
            raise SystemExit(1)

    for i, (method, phase, runner) in enumerate(plan, start=1):
        step_started_at = print_step_banner(i, total_steps, method, phase)
        step = runner()
        record(step)
        print_step_footer(i, total_steps, step, step_started_at, ok_count)

    merged_path = merge_summaries(run_root, steps)
    append_text(index_path, f"\n[MERGED] summary_all_methods.csv: {abspath(merged_path)}\n")

    total_elapsed = time.time() - overall_started_at
    print("\n" + "=" * 88)
    print("ALL DONE")
    print(f"Run root:     {abspath(run_root)}")
    print(f"Index:        {abspath(index_path)}")
    print(f"Merged:       {abspath(merged_path)}")
    print(f"Total time:   {fmt_seconds(total_elapsed)}")
    print(f"Step success: {ok_count}/{total_steps}")
    print("=" * 88)


if __name__ == "__main__":
    main()
    # Example:
    # python .\run_all_final_progress.py
