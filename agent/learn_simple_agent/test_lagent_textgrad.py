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
os.environ["DEEPSEEK_HOST"] = "api.deepseek.com"
os.environ["DEEPSEEK_KEY"] = "sk-457d831f3fd24603ad514f9f26dbe132"
os.environ["DEEPSEEK_MODEL"] = "deepseek-chat"

os.environ["QWEN_HOST"] = "dashscope.aliyuncs.com"
os.environ["QWEN_KEY"] = "sk-f622faad8dee4b179d1c8593b1dab866"
os.environ["QWEN_MODEL"] = "qwen-flash"


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
    
def _generate_single_model_desc(
    teacher_llm: HttpsApi,
    role_hint: str,
    fallback_desc: str,
) -> str:
    """
    为“一个匿名 backend LLM”生成一段能力描述。
    - 不提任何具体模型名字（DeepSeek / Qwen / GPT 等）。
    - 只描述它在 AHD/OBP 场景下可能的行为特征。
    - 若解析失败，则回退到给定的 fallback_desc。
    """

    system_prompt = (
        "You are proposing hypothetical capability profiles for anonymous backend LLMs "
        "used in an automatic heuristic design (AHD) system based on Evolution of "
        "Heuristics (EoH) for online bin packing."
    )

    user_prompt = f"""
We need a NEW capability description for ONE anonymous backend LLM.

Context:
- This model will be one of several backends used by a routing agent.
- The routing agent will NOT know the true name or architecture of the model.
- It will only see the natural-language description you write here.

Your task for this particular description :
- Invent plausible hypotheses about where THIS model tends to be strong or weak
  when generating or editing heuristic code for combinatorial optimization
  (e.g., online bin packing).
- Focus on BEHAVIOURAL properties.
- Explore the non-traditional advantages as much as possible. Use your imagination to 
    the fullest to consider what unconventional strengths or weaknesses this LLM might possess, 
    exploring from unexpected angles.

STRICT constraints:
- Treat this as a single anonymous model. You do NOT know its vendor or architecture.
- DO NOT mention any concrete model names (like "DeepSeek", "Qwen", "GPT", "Claude", etc.).
- DO NOT compare it to other models. Just describe THIS model ("this model", "it", etc.).
- Do not talk about API price, latency, or quota.

Output format:
Return ONLY a JSON object with the following field:

{{
  "desc": "a markdown bullet list (4-7 bullets) describing this model's strengths and weaknesses"
}}
"""

    try:
        raw = teacher_llm.draw_sample(prompt=f"{system_prompt}\n\n{user_prompt}")
    except TypeError:
        raw = teacher_llm.draw_sample(f"{system_prompt}\n\n{user_prompt}")

    parsed = _extract_json_block(raw) or {}

    # 如果 teacher 返回的是一个 list，比如 [{"desc": "..."}]，也处理一下
    if isinstance(parsed, list):
        if parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            parsed = {}

    value = parsed.get("desc", "")

    # 统一把 value 变成字符串
    if isinstance(value, list):
        # teacher 可能返回 ["- bullet1", "- bullet2"]
        desc = "\n".join(str(x) for x in value)
    else:
        desc = str(value)

    desc = desc.strip()
    if not desc:
        # 解析失败或为空，则退回默认描述
        desc = fallback_desc

    return desc

def generate_initial_population(
    teacher_llm: HttpsApi,
    pop_size: int,
    include_baseline: bool = True,
) -> List[RouterPromptCandidate]:
    """
    用一个较强的 teacher LLM 自动生成一批“模型能力描述”，
    但每个模型的描述是单独、匿名地生成的：
    - 不在 teacher prompt 中提到 DeepSeek / Qwen 等具体名字；
    - 不强制要求两个模型互补；
    - 只让 teacher 为“某个匿名 backend LLM”写出一组行为假设。
    最后仍然用 deepseek_desc / qwen_desc 两个字段来存放这两段文字。
    """

    candidates: List[RouterPromptCandidate] = []
    cid = 0

    # 可选：把当前手写的默认描述作为 baseline（对照）
    if include_baseline:
        candidates.append(
            RouterPromptCandidate(
                cid=cid,
                deepseek_desc=DEFAULT_DEEPSEEK_DESC,
                qwen_desc=DEFAULT_QWEN_DESC,
            )
        )
        cid += 1

    # 后续 candidate：对“模型1 / 模型2”分别生成匿名描述
    while len(candidates) < pop_size:
        # 这里的 role_hint 只是给 teacher 一点轻微的多样性提示，
        # 不会泄露实际模型名字
        deepseek_desc = _generate_single_model_desc(
            teacher_llm=teacher_llm,
            role_hint="",
            fallback_desc=DEFAULT_DEEPSEEK_DESC,
        )

        qwen_desc = _generate_single_model_desc(
            teacher_llm=teacher_llm,
            role_hint="",
            fallback_desc=DEFAULT_QWEN_DESC,
        )

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


def _update_single_model_desc(
    teacher_llm: HttpsApi,
    best_score: float,
    weak_score: float,
    best_desc: str,
    weak_desc: str,
    slot_hint: str,
) -> str:
    """
    用 TextGrad / TPD 思路，针对“某一个匿名 backend 槽位”做一次文本梯度更新：
    - 输入：Router A/B 在 *同一个 backend 槽位* 上的描述 + 两个 router 的分数
    - 输出：更新后的 weak_desc（保持匿名）

    这里不提 DeepSeek/Qwen/GPT 等真实名字，只说“this backend model”。
    """

    system_prompt = (
        "You are optimizing the textual capability description for an anonymous backend LLM "
        "used inside a routing agent in an automatic heuristic design (AHD) system based on "
        "Evolution of Heuristics (EoH) for an online bin packing task."
    )

    # ===== debug: 输入信息 =====
    print("\n" + "=" * 80)
    print("[_update_single_model_desc] Start TextGrad update")
    print(f"  slot_hint   : {slot_hint}")
    print(f"  best_score  : {best_score}")
    print(f"  weak_score  : {weak_score}")
    print("  Router A desc (best):")
    print(best_desc)
    print("-" * 60)
    print("  Router B desc (weak):")
    print(weak_desc)
    print("=" * 80)

    user_prompt = f"""
We have two router prompts:

- Router A (better): average best_score = {best_score}
- Router B (worse): average best_score = {weak_score}

They only differ in how they describe ONE particular backend slot in the router:
{slot_hint}

For THIS backend slot, the two router prompts use different descriptions
for the SAME anonymous backend model.

Router A's description for this backend:
----------------------------------------
{best_desc}
----------------------------------------

Router B's description for this backend:
----------------------------------------
{weak_desc}
----------------------------------------

You do NOT know the true vendor, architecture, or real-world name of this model.
It is just "this backend model" in the routing system.

Step 1: Textual loss (analysis)
- Briefly compare Router A's and Router B's descriptions for THIS backend.
- Explain why Router A's description is more likely to lead to better routing decisions
  in the AHD/online-bin-packing setting. Focus on behavioural properties such as:
  * when this backend should be chosen,
  * what kinds of prompts / operators it is suited for,
  * what important capabilities or limitations are clearly stated or missing.

Step 2: Textual gradient (update instructions)
- Based on the above comparison, propose concrete update instructions for IMPROVING
  ONLY Router B's description for THIS backend model.
- The goal is to make the description:
  * more specific and predictive of when this backend will work well or poorly,
  * clearer about the kinds of heuristic-generation or code-editing behaviour it excels at,
  * better aligned with the high-level demands of EoH and online bin packing.
- You do NOT need to enforce complementarity with other backends in the system.
  Focus on making THIS model's description internally coherent and useful for routing.

Step 3: Updated description
- Apply your own update instructions.
- Write a NEW, improved description for this backend model, in the same style
  (a short markdown bullet list of 4–7 bullets).

STRICT constraints:
- Do NOT mention any concrete model names (like "DeepSeek", "Qwen", "GPT", "Claude", etc.).
- Do NOT talk about API price, latency, or quota.
- Do NOT assume knowledge about other backends; optimize THIS backend description alone.

Return ONLY a JSON object with the following fields:

{{
  "updated_desc": "updated markdown bullet list for this backend model",
  "analysis": "optional 3-8 sentence explanation of the main changes"
}}
"""

    try:
        raw = teacher_llm.draw_sample(prompt=f"{system_prompt}\n\n{user_prompt}")
    except TypeError:
        raw = teacher_llm.draw_sample(f"{system_prompt}\n\n{user_prompt}")

    # ===== debug: LLM 原始输出（截断） =====
    raw_str = str(raw)
    print("[_update_single_model_desc] Raw LLM output (truncated to 800 chars):")
    print(raw_str[:800])
    if len(raw_str) > 800:
        print("... [truncated] ...")

    parsed = _extract_json_block(raw) or {}

    # 兼容几种常见乱写格式：list / dict / 字符串
    if isinstance(parsed, list):
        if parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            parsed = {}

    print("[_update_single_model_desc] Parsed JSON:")
    print(parsed)

    value = parsed.get("updated_desc") or parsed.get("desc") or ""

    # 有些模型会给 list，当成多行 bullet；有些给 str
    if isinstance(value, list):
        new_desc = "\n".join(str(x) for x in value)
    else:
        new_desc = str(value)

    new_desc = new_desc.strip()
    if not new_desc:
        # 解析失败就回退到原始 weak_desc
        print("[_update_single_model_desc] Empty new_desc, fallback to weak_desc.")
        new_desc = weak_desc
    else:
        print("[_update_single_model_desc] Updated description for this backend:")
        print(new_desc)

    print("[_update_single_model_desc] End TextGrad update")
    print("=" * 80 + "\n")

    return new_desc



# ===================== 反向：文本梯度更新 prompt =====================
def update_candidate_with_textgrad(
    teacher_llm: HttpsApi,
    best_cand: RouterPromptCandidate,
    weak_cand: RouterPromptCandidate,
) -> RouterPromptCandidate:
    """
    使用 TextGrad / TPD 思路，对“较差 router (weak_cand)”做文本梯度更新。

    与旧版的区别：
    1. 匿名化：teacher 的 prompt 中不再出现 DeepSeek/Qwen 等真实名字，
       只说“某个 backend 槽位上的匿名模型”。
    2. 单独更新：对每个 backend 槽位（slot #1 / slot #2）分别调用一次
       _update_single_model_desc，只更新对应的一段描述，而不是让 teacher
       在同一个 prompt 里同时改两段描述。

    返回：一个新的 RouterPromptCandidate，仅更新 weak_cand 的描述内容，
    id 保持不变，score/meta 置空（下一轮再重新评估）。
    """
    
    # 防御：如果没有分数，就当 0 处理（一般不会出现）
    best_score = best_cand.score if best_cand.score is not None else 0.0
    weak_score = weak_cand.score if weak_cand.score is not None else 0.0

    # 1) 更新“backend slot #1”的描述（代码里映射到 deepseek_desc 槽位）
    updated_desc_slot1 = _update_single_model_desc(
        teacher_llm=teacher_llm,
        best_score=best_score,
        weak_score=weak_score,
        best_desc=best_cand.deepseek_desc,
        weak_desc=weak_cand.deepseek_desc,
        slot_hint="backend slot #1 (the first anonymous LLM option in the router)",
    )

    # 2) 更新“backend slot #2”的描述（代码里映射到 qwen_desc 槽位）
    updated_desc_slot2 = _update_single_model_desc(
        teacher_llm=teacher_llm,
        best_score=best_score,
        weak_score=weak_score,
        best_desc=best_cand.qwen_desc,
        weak_desc=weak_cand.qwen_desc,
        slot_hint="backend slot #2 (the second anonymous LLM option in the router)",
    )

    return RouterPromptCandidate(
        cid=weak_cand.cid,  # 保留同一个 id，只更新内容
        deepseek_desc=updated_desc_slot1,
        qwen_desc=updated_desc_slot2,
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
    for individual in population:
        print(individual)


    # 3. 迭代优化
    best_overall: Optional[RouterPromptCandidate] = None

    for g in range(num_generations):
        print("\n" + "=" * 80)
        print(f"Generation {g+1}/{num_generations}")
        print("=" * 80)

        gen_log_root = os.path.join(log_root, f"gen_{g+1}")
        os.makedirs(gen_log_root, exist_ok=True)

        # 3.1 评估所有 candidate
        evaluated: List[RouterPromptCandidate] = []
        for cand in population:
            print("CAND！！！！",cand)
            eval_cand = evaluate_candidate(
                candidate=cand,
                deepseek_llm=deepseek_llm,
                qwen_llm=qwen_llm,
                router_agent_llm=router_agent_llm,
                log_root=gen_log_root,
                num_runs=num_runs_per_candidate,
            )
            print("!!!!end CAND")
            evaluated.append(eval_cand)

        # 3.2 排序，分数越大越好
        evaluated = [c for c in evaluated if c.score is not None]
        if not evaluated:
            raise RuntimeError("All candidates failed to obtain a valid score.")

        evaluated.sort(key=lambda c: c.score, reverse=True)
        best = evaluated[0]

        print(f"\n[Generation {g+1}] Best candidate id={best.cid}, score={best.score}")

        # 更新全局 best
        if best_overall is None or best.score > best_overall.score:
            best_overall = best

        # 保存当前种群信息到 json，方便之后分析
        snapshot = {
            "generation": g + 1,
            "candidates": [asdict(c) for c in evaluated],
            "timestamp": time.time(),
        }
        with open(os.path.join(gen_log_root, "population.json"), "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        # 最后一代就不用再反向更新了
        if g == num_generations - 1:
            break

        # 3.3 TPD 风格：以 best 为锚点，更新其余个体
        new_population: List[RouterPromptCandidate] = [best]  # 精英保留
        for weak in evaluated[1:population_size]:
            updated = update_candidate_with_textgrad(
                teacher_llm=teacher_llm,
                best_cand=best,
                weak_cand=weak,
            )
            new_population.append(updated)

        # 下一代用更新后的种群（如果 population_size 比 evaluated 小，会自动截断）
        population = new_population[:population_size]

    assert best_overall is not None
    print("\n========== Finished router prompt learning ==========")
    print(f"Best overall candidate id={best_overall.cid}, score={best_overall.score}")

    # 把最终最优的描述单独存成一个 json，方便后续直接加载
    final_path = os.path.join(log_root, "best_router_prompt.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cid": best_overall.cid,
                "deepseek_desc": best_overall.deepseek_desc,
                "qwen_desc": best_overall.qwen_desc,
                "score": best_overall.score,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved best router prompt to: {final_path}")

    return best_overall

optimize_router_prompts(num_generations=3,
population_size=5,
num_runs_per_candidate=3,
log_root="logs_router_learning_11280248")