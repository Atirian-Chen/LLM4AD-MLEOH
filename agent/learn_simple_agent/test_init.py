# router_prompt_learning.py
"""
使用 TextGrad + TPD-AHD 思想，为 AgentBasedRouter 学习一段更好的
“模型能力描述”（只改写 prompt 中 [DEEPSEEK]/[QWEN] 那部分）。

整体流程（对应 TPD-AHD 的 forward / backward）：

1. 初始化一组 router-prompt 候选（population），变量是
   - deepseek_desc
   - qwen_desc

2. 对每个候选：
   - 用对应的描述实例化 AgentBasedRouter
   - 在 OBP + EoH 上跑一小轮，拿到 best_score 作为 fitness
   （EoH 的 fitness 定义与原论文一致：越大越好）

3. 排序得到当前最优候选 h1，构造 best-anchored preference pairs：
   (h1, h2), (h1, h3), ... (h1, hN)，对应 TPD-AHD 中的偏好配对机制。

4. 对每一对 (h1, hk)，调用一个“teacher LLM”：
   - 比较两个 router prompt 的描述 + 分数，生成 textual loss（解释为何 h1 更好）。
   - 再生成 textual gradient：给出“如何修改 hk 的 deepseek_desc / qwen_desc 才能更接近 h1，
     同时保留/探索一些新的假设维度”。这一点对应 TextGrad 中通过自然语言反馈
     回传梯度信号的思想。

5. 用 textual gradient 直接让 LLM 产出新的 (deepseek_desc', qwen_desc')，形成下一代 population。

6. 重复若干代，最后选出平均性能最好的 router prompt 作为论文中的“学习型 router”。
"""

import json
import os
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

from lagent_router import AgentBasedRouter, DEFAULT_DEEPSEEK_DESC, DEFAULT_QWEN_DESC


# ===================== 数据结构 =====================

@dataclass
class RouterPromptCandidate:
    cid: int
    deepseek_desc: str
    qwen_desc: str
    # 评估得到的 EoH best_score（越大越好）
    score: Optional[float] = None
    # 额外统计信息（调用次数等）
    meta: Optional[Dict[str, Any]] = None


# ===================== 一些小工具 =====================

def build_https_llm_from_env(prefix: str) -> HttpsApi:
    """
    根据环境变量构造一个 HttpsApi：
    - {PREFIX}_HOST
    - {PREFIX}_KEY
    - {PREFIX}_MODEL

    例如：
    - DEEPSEEK_HOST, DEEPSEEK_KEY, DEEPSEEK_MODEL
    - QWEN_HOST, QWEN_KEY, QWEN_MODEL
    - ROUTER_HOST, ROUTER_KEY, ROUTER_MODEL
    - TEACHER_HOST, TEACHER_KEY, TEACHER_MODEL
    """
    host = os.environ.get(f"{prefix}_HOST")
    key = os.environ.get(f"{prefix}_KEY")
    model = os.environ.get(f"{prefix}_MODEL")

    if host is None or key is None or model is None:
        raise ValueError(
            f"Missing env vars for {prefix}: "
            f"{prefix}_HOST / {prefix}_KEY / {prefix}_MODEL"
        )

    timeout = int(os.environ.get(f"{prefix}_TIMEOUT", "60"))

    return HttpsApi(
        host=host,
        key=key,
        model=model,
        timeout=timeout,
    )


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM 输出中粗暴地抓取第一个 JSON 块并解析。
    如果失败就返回 None，由调用方做 fallback。
    """
    if not text:
        return None

    s = str(text)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = s[start : end + 1]
    try:
        return json.loads(json_str)
    except Exception:
        return None


# ===================== 初始化：随机 prompt 种群 =====================

def generate_initial_population(
    teacher_llm: HttpsApi,
    pop_size: int,
    include_baseline: bool = True,
) -> List[RouterPromptCandidate]:
    """
    用一个较强的 teacher LLM 自动生成一批“模型能力描述”，
    尽量覆盖不同的潜在维度（例如：全局规划、局部修补、长上下文跟踪、
    数值稳定性、边界情况处理、风格一致性等等），而不局限在“创意/细心/代码能力强”。
    """
    candidates: List[RouterPromptCandidate] = []
    cid = 0

    if include_baseline:
        # 把你当前使用的静态 router 描述作为一个对照基线
        candidates.append(
            RouterPromptCandidate(
                cid=cid,
                deepseek_desc=DEFAULT_DEEPSEEK_DESC,
                qwen_desc=DEFAULT_QWEN_DESC,
            )
        )
        cid += 1

    system_prompt = (
        "You are designing routing prompts for an automatic heuristic design (AHD) "
        "system based on Evolution of Heuristics (EoH) for online bin packing. "
        "Two backend LLMs are available: DEEPSEEK and QWEN. "
        "You will propose diverse hypotheses about their relative strengths and weaknesses."
    )

    for _ in range(pop_size - len(candidates)):
        user_prompt = f"""
We need a NEW pair of capability descriptions for two LLMs, [DEEPSEEK] and [QWEN],
used inside a routing agent. The agent will see the full AHD prompt (including task
description, operator info like E1/E2/M1/M2, current heuristic code, etc.) and must
choose which backend model is more suitable for THIS call.

Requirements for this particular candidate:
- Explore some *less obvious* capability dimensions (e.g., robustness to noisy code,
  ability to reason about capacity constraints, sensitivity to long-term state, etc.).
- Make DEEPSEEK and QWEN *complementary* rather than symmetric.
- Do NOT talk about API quota, price, or latency.
- Each description should be 4–7 bullet points, short but concrete.

Return ONLY a JSON object with the following fields:

{{
  "deepseek_desc": "markdown bullet list describing DEEPSEEK's strengths/weaknesses",
  "qwen_desc": "markdown bullet list describing QWEN's strengths/weaknesses"
}}
"""
        try:
            raw = teacher_llm.draw_sample(prompt=f"{system_prompt}\n\n{user_prompt}")
        except TypeError:
            raw = teacher_llm.draw_sample(f"{system_prompt}\n\n{user_prompt}")

        parsed = _extract_json_block(raw) or {}
        deepseek_desc = parsed.get("deepseek_desc", DEFAULT_DEEPSEEK_DESC)
        qwen_desc = parsed.get("qwen_desc", DEFAULT_QWEN_DESC)

        candidates.append(
            RouterPromptCandidate(
                cid=cid,
                deepseek_desc=deepseek_desc,
                qwen_desc=qwen_desc,
            )
        )
        cid += 1

    return candidates


# ===================== 前向：给一个 candidate 评估一次 =====================

def evaluate_candidate_once(
    candidate: RouterPromptCandidate,
    run_id: int,
    deepseek_llm: HttpsApi,
    qwen_llm: HttpsApi,
    router_agent_llm: HttpsApi,
    log_root: str,
    pop_size: int = 8,
    max_generations: int = 6,
    max_sample_nums: int = 10000,
) -> Dict[str, Any]:
    """
    基本就是你原来的 multirun_eoh_agent_router.py，只是把
    AgentBasedRouter 的描述替换成 candidate 的 desc。
    返回当前 run 的 best_score 和调用统计。
    """

    print(f"\n========== [Candidate {candidate.cid}] Run {run_id} ==========")

    # 1. 把三个 LLM client 拼成一个 router
    router_llm = AgentBasedRouter(
        deepseek_llm=deepseek_llm,
        qwen_llm=qwen_llm,
        router_llm=router_agent_llm,
        deepseek_desc=candidate.deepseek_desc,
        qwen_desc=candidate.qwen_desc,
        verbose=True,
    )

    # 2. 任务和 profiler
    task = OBPEvaluation()
    # 如果需要区分种子，可以按 candidate + run_id 改种子
    # task.random_seed = run_id + candidate.cid * 1000

    log_dir = os.path.join(
        log_root,
        f"candidate_{candidate.cid}",
        f"run_{run_id}",
    )
    os.makedirs(log_dir, exist_ok=True)
    profiler = EoHProfiler(
        log_dir=log_dir,
        log_style="simple",
    )

    # 3. EoH 配置（可以适当减小以节省学习时间）
    method = EoH(
        llm=router_llm,
        evaluation=task,
        profiler=profiler,
        pop_size=pop_size,
        max_generations=max_generations,
        max_sample_nums=max_sample_nums,
        selection_num=3,
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,
        num_samplers=4,
        num_evaluators=4,
        debug_mode=False,
    )

    # 4. 运行
    print(">>> Start EoH (Agent Router + DeepSeek + Qwen) on Online Bin Packing...")
    _ = method.run()
    print(">>> Search finished.")

    best_score = method.best_score  # EoH 定义：越大越好（等价于 -objective）

    total_calls = router_llm.call_cnt
    deepseek_calls = router_llm.deepseek_cnt
    qwen_calls = router_llm.qwen_cnt

    print(f"[Candidate {candidate.cid}] Run {run_id} best_score = {best_score}")
    print(
        f"[Candidate {candidate.cid}] calls: "
        f"total={total_calls}, deepseek={deepseek_calls}, qwen={qwen_calls}"
    )
    if total_calls > 0:
        print(
            f"[Candidate {candidate.cid}] ratio: "
            f"deepseek={deepseek_calls / total_calls:.3f}, "
            f"qwen={qwen_calls / total_calls:.3f}"
        )

    # EoH.run() 内部会调用 router_llm.close()，会重复打印一次统计，这是正常的。

    return {
        "best_score": best_score,
        "total_calls": total_calls,
        "deepseek_calls": deepseek_calls,
        "qwen_calls": qwen_calls,
    }


def evaluate_candidate(
    candidate: RouterPromptCandidate,
    deepseek_llm: HttpsApi,
    qwen_llm: HttpsApi,
    router_agent_llm: HttpsApi,
    log_root: str,
    num_runs: int = 3,
    **eoh_kwargs,
) -> RouterPromptCandidate:
    """
    对同一个 candidate 跑 num_runs 次 EoH，取平均 best_score。
    返回一个带 score / meta 的新 candidate（不 in-place 修改原对象，方便追踪）。
    """
    scores: List[float] = []
    run_metas: List[Dict[str, Any]] = []

    for r in range(1, num_runs + 1):
        result = evaluate_candidate_once(
            candidate=candidate,
            run_id=r,
            deepseek_llm=deepseek_llm,
            qwen_llm=qwen_llm,
            router_agent_llm=router_agent_llm,
            log_root=log_root,
            **eoh_kwargs,
        )
        if result["best_score"] is not None:
            scores.append(result["best_score"])
        run_metas.append(result)

    if scores:
        avg_score = statistics.mean(scores)
    else:
        avg_score = None

    meta = {
        "runs": run_metas,
        "avg_score": avg_score,
    }

    return RouterPromptCandidate(
        cid=candidate.cid,
        deepseek_desc=candidate.deepseek_desc,
        qwen_desc=candidate.qwen_desc,
        score=avg_score,
        meta=meta,
    )


# ===================== 反向：文本梯度更新 prompt =====================

def update_candidate_with_textgrad(
    teacher_llm: HttpsApi,
    best_cand: RouterPromptCandidate,
    weak_cand: RouterPromptCandidate,
) -> RouterPromptCandidate:
    """
    构造一个 TextGrad / TPD 风格的 prompt，让 teacher LLM：
    - 对比 best_cand 和 weak_cand 的描述 + 分数
    - 生成 textual loss：解释为什么 best 更好
    - 再生成 textual “梯度”：如何修改 weak_cand 的两段描述

    最终直接输出一个新的 (deepseek_desc, qwen_desc)。
    """

    system_prompt = (
        "You are optimizing the routing prompt of a compound AI system. "
        "The system uses a routing agent to choose between two backend LLMs "
        "for automatic heuristic design (EoH) on an online bin packing task. "
        "Your job is to refine ONLY the capability descriptions of the two models "
        "([DEEPSEEK] and [QWEN]) so that the router chooses better backends."
    )

    user_prompt = f"""
We have two router prompts (A = better, B = worse). They only differ in the way they
describe the capabilities of the two backend models.

Each router prompt was evaluated inside the same EoH + OBP pipeline.
Higher best_score means better heuristics.

Router A (better):
- average best_score: {best_cand.score}
- [DEEPSEEK] description:
{best_cand.deepseek_desc}

- [QWEN] description:
{best_cand.qwen_desc}

Router B (worse):
- average best_score: {weak_cand.score}
- [DEEPSEEK] description:
{weak_cand.deepseek_desc}

- [QWEN] description:
{weak_cand.qwen_desc}

Step 1: Textual loss (analysis)
Briefly compare Router A and B. Explain:
- In what kinds of prompts / operator patterns Router A is likely to make better routing decisions.
- Which parts of Router B's descriptions are vague, misleading, redundant, or missing important hypotheses.

Step 2: Textual gradient (update instructions)
Based on the above comparison, propose concrete update instructions for Router B's descriptions ONLY.
Focus on:
- Clarifying when DEEPSEEK vs QWEN should be preferred for this AHD/OBP task.
- Surfacing subtle but important capability dimensions (e.g., reasoning about capacity constraints,
  handling long chains of code modifications, robustness to slightly buggy code, etc.).
- Keeping the two models complementary (avoid saying they are good at exactly the same thing).

Step 3: Updated descriptions
Apply your own update instructions and output a NEW pair of descriptions for Router B:

Return ONLY a JSON object with the following fields:

{{
  "deepseek_desc": "updated markdown bullet list for DEEPSEEK",
  "qwen_desc": "updated markdown bullet list for QWEN",
  "analysis": "optional 3-8 sentence explanation of the main changes"
}}
"""

    try:
        raw = teacher_llm.draw_sample(prompt=f"{system_prompt}\n\n{user_prompt}")
    except TypeError:
        raw = teacher_llm.draw_sample(f"{system_prompt}\n\n{user_prompt}")

    parsed = _extract_json_block(raw) or {}
    new_deepseek = parsed.get("deepseek_desc", weak_cand.deepseek_desc)
    new_qwen = parsed.get("qwen_desc", weak_cand.qwen_desc)

    return RouterPromptCandidate(
        cid=weak_cand.cid,  # 保留同一个 id，只更新内容
        deepseek_desc=new_deepseek,
        qwen_desc=new_qwen,
        score=None,
        meta=None,
    )


# ===================== 总体优化循环 =====================

def optimize_router_prompts(
    num_generations: int = 3,
    population_size: int = 4,
    num_runs_per_candidate: int = 2,
    log_root: str = "logs_router_learning",
) -> RouterPromptCandidate:
    """
    整体学习循环：
    - 初始化一批 candidate
    - 每一代：
        * 前向：在 EoH+OBP 上评估所有 candidate
        * 构造 best-anchored pairs (h_best, h_i)
        * 反向：用 textual gradient 更新所有非最优个体
    - 返回最后一代中分数最高的 candidate
    """

    # 1. 构造几个 LLM client
    deepseek_llm = build_https_llm_from_env("DEEPSEEK")
    qwen_llm = build_https_llm_from_env("QWEN")

    # 路由 agent 可以和其中一个共用，也可以单独指定 ROUTER_*
    try:
        router_agent_llm = build_https_llm_from_env("ROUTER")
    except ValueError:
        router_agent_llm = qwen_llm

    # teacher LLM 用于生成 / 更新 prompt，一般选择最强的模型
    try:
        teacher_llm = build_https_llm_from_env("TEACHER")
    except ValueError:
        teacher_llm = router_agent_llm

    os.makedirs(log_root, exist_ok=True)

    # 2. 初始化种群
    population = generate_initial_population(
        teacher_llm=teacher_llm,
        pop_size=population_size,
        include_baseline=True,
    )

    # # 3. 迭代优化
    # best_overall: Optional[RouterPromptCandidate] = None

    # for g in range(num_generations):
    #     print("\n" + "=" * 80)
    #     print(f"Generation {g+1}/{num_generations}")
    #     print("=" * 80)

    #     gen_log_root = os.path.join(log_root, f"gen_{g+1}")
    #     os.makedirs(gen_log_root, exist_ok=True)

    #     # 3.1 评估所有 candidate
    #     evaluated: List[RouterPromptCandidate] = []
    #     for cand in population:
    #         eval_cand = evaluate_candidate(
    #             candidate=cand,
    #             deepseek_llm=deepseek_llm,
    #             qwen_llm=qwen_llm,
    #             router_agent_llm=router_agent_llm,
    #             log_root=gen_log_root,
    #             num_runs=num_runs_per_candidate,
    #         )
    #         evaluated.append(eval_cand)

    #     # 3.2 排序，分数越大越好
    #     evaluated = [c for c in evaluated if c.score is not None]
    #     if not evaluated:
    #         raise RuntimeError("All candidates failed to obtain a valid score.")

    #     evaluated.sort(key=lambda c: c.score, reverse=True)
    #     best = evaluated[0]

    #     print(f"\n[Generation {g+1}] Best candidate id={best.cid}, score={best.score}")

    #     # 更新全局 best
    #     if best_overall is None or best.score > best_overall.score:
    #         best_overall = best

    #     # 保存当前种群信息到 json，方便之后分析
    #     snapshot = {
    #         "generation": g + 1,
    #         "candidates": [asdict(c) for c in evaluated],
    #         "timestamp": time.time(),
    #     }
    #     with open(os.path.join(gen_log_root, "population.json"), "w", encoding="utf-8") as f:
    #         json.dump(snapshot, f, ensure_ascii=False, indent=2)

    #     # 最后一代就不用再反向更新了
    #     if g == num_generations - 1:
    #         break

    #     # 3.3 TPD 风格：以 best 为锚点，更新其余个体
    #     new_population: List[RouterPromptCandidate] = [best]  # 精英保留
    #     for weak in evaluated[1:population_size]:
    #         updated = update_candidate_with_textgrad(
    #             teacher_llm=teacher_llm,
    #             best_cand=best,
    #             weak_cand=weak,
    #         )
    #         new_population.append(updated)

    #     # 下一代用更新后的种群（如果 population_size 比 evaluated 小，会自动截断）
    #     population = new_population[:population_size]

    # assert best_overall is not None
    # print("\n========== Finished router prompt learning ==========")
    # print(f"Best overall candidate id={best_overall.cid}, score={best_overall.score}")

    # # 把最终最优的描述单独存成一个 json，方便后续直接加载
    # final_path = os.path.join(log_root, "best_router_prompt.json")
    # with open(final_path, "w", encoding="utf-8") as f:
    #     json.dump(
    #         {
    #             "cid": best_overall.cid,
    #             "deepseek_desc": best_overall.deepseek_desc,
    #             "qwen_desc": best_overall.qwen_desc,
    #             "score": best_overall.score,
    #         },
    #         f,
    #         ensure_ascii=False,
    #         indent=2,
    #     )
    # print(f"Saved best router prompt to: {final_path}")

    return population


# ===================== 一个简单的测试入口 =====================

def test_learned_router(
    prompt_json_path: str,
    num_runs: int = 5,
    log_root: str = "logs_router_learned_eval",
):
    """
    给定已经学到的 best_router_prompt.json，重复多次 EoH，验证性能。

    这个函数基本就是你原来的 multirun_eoh_agent_router.py，
    只是从文件里读取 deepseek_desc / qwen_desc。
    """
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    deepseek_desc = data["deepseek_desc"]
    qwen_desc = data["qwen_desc"]

    deepseek_llm = build_https_llm_from_env("DEEPSEEK")
    qwen_llm = build_https_llm_from_env("QWEN")
    try:
        router_agent_llm = build_https_llm_from_env("ROUTER")
    except ValueError:
        router_agent_llm = qwen_llm

    all_scores: List[float] = []
    all_results: List[Dict[str, Any]] = []

    for r in range(1, num_runs + 1):
        cand = RouterPromptCandidate(
            cid=0,
            deepseek_desc=deepseek_desc,
            qwen_desc=qwen_desc,
        )
        result_cand = evaluate_candidate(
            candidate=cand,
            deepseek_llm=deepseek_llm,
            qwen_llm=qwen_llm,
            router_agent_llm=router_agent_llm,
            log_root=os.path.join(log_root, f"run_{r}"),
            num_runs=1,
        )
        all_results.extend(result_cand.meta["runs"])
        if result_cand.score is not None:
            all_scores.append(result_cand.score)

    if all_scores:
        avg_best_score = statistics.mean(all_scores)
    else:
        avg_best_score = None

    total_deepseek = sum(r["deepseek_calls"] for r in all_results)
    total_qwen = sum(r["qwen_calls"] for r in all_results)
    total_calls = total_deepseek + total_qwen

    if total_calls > 0:
        avg_deepseek_ratio = total_deepseek / total_calls
        avg_qwen_ratio = total_qwen / total_calls
    else:
        avg_deepseek_ratio = 0.0
        avg_qwen_ratio = 0.0

    print("\n========== Summary over learned-router runs ==========")
    print(f"Number of runs: {num_runs}")
    print(f"Average best_score: {avg_best_score}")
    print(
        "Average call ratio (aggregated over all runs): "
        f"deepseek={avg_deepseek_ratio:.3f}, qwen={avg_qwen_ratio:.3f}"
    )



best = optimize_router_prompts(
        num_generations=2,
        population_size=4,
        num_runs_per_candidate=1,
        log_root="logs_router_learning_demo",
    )
    
for individual in best:
    print(individual)
    # print("\n>>> Now evaluating the learned router on a few fresh runs...")
    # test_learned_router(
    #     prompt_json_path="logs_router_learning_demo/best_router_prompt.json",
    #     num_runs=3,
    #     log_root="logs_router_learned_eval_demo",
    # )
