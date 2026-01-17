# test_eoh_obp_router.py
import os
import statistics

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

from agent_router import AgentBasedRouter


# 你想跑多少轮实验
N_RUNS = 5


def run_one_experiment(run_id: int):
    """
    跑一轮 EoH + Router 实验，返回本轮的 best_score 和调用统计。
    """

    print(f"\n========== Run {run_id}/{N_RUNS} ==========")

    # ===== 1. 配置三个 LLM client =====
    DEEPSEEK_API_KEY = "sk-457d831f3fd24603ad514f9f26dbe132"
    QWEN_API_KEY = "sk-f622faad8dee4b179d1c8593b1dab866"

    # 1.1 DeepSeek 实例
    deepseek_llm = HttpsApi(
        host="api.deepseek.com",
        key=DEEPSEEK_API_KEY,
        model="deepseek-chat",
        timeout=60,
    )

    # 1.2 Qwen 实例（阿里 DashScope）
    qwen_llm = HttpsApi(
        host="dashscope.aliyuncs.com",
        key=QWEN_API_KEY,
        model="qwen-flash",
        timeout=60,
    )

    # 路由 agent：这里还是用 Qwen，当 router
    router_agent_llm = deepseek_llm

    # 把三个拼成一个逻辑上的 “llm”
    router_llm = AgentBasedRouter(
        deepseek_llm=deepseek_llm,
        qwen_llm=qwen_llm,
        router_llm=router_agent_llm,
        verbose=True,
    )

    # ===== 2. 任务和 profiler =====
    task = OBPEvaluation()
    # 如果你想每轮换随机种子，可以这样（看 OBPEvaluation 是否用这个字段）:
    # task.random_seed = run_id

    log_dir = os.path.join("logs", "eoh_obp_router_agent", f"run_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    profiler = EoHProfiler(
        log_dir=log_dir,
        log_style="simple",
    )

    # ===== 3. EoH 配置 =====
    method = EoH(
        llm=router_llm,          # << 用 router 代替原来的 HttpsApi
        evaluation=task,
        profiler=profiler,

        pop_size=8,
        max_generations=8,
        max_sample_nums=20000,

        selection_num=3,
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,

        num_samplers=4,
        num_evaluators=4,

        debug_mode=False,
    )

    # ===== 4. 运行 =====
    print(">>> Start EoH (Agent Router + DeepSeek + Qwen) on Online Bin Packing...")
    best_program = method.run()
    print(">>> Search finished.")

    best_score = method.best_score

    # 从 router_llm 里拿统计信息
    total_calls = router_llm.call_cnt
    deepseek_calls = router_llm.deepseek_cnt
    qwen_calls = router_llm.qwen_cnt

    # 本轮结果打印一下
    print(f"Run {run_id} best_score = {best_score}")
    print(f"Run {run_id} calls: total={total_calls}, deepseek={deepseek_calls}, qwen={qwen_calls}")
    if total_calls > 0:
        print(
            f"Run {run_id} ratio: "
            f"deepseek={deepseek_calls / total_calls:.3f}, "
            f"qwen={qwen_calls / total_calls:.3f}"
        )

    # 注意：EoH.run() 里会调用 router_llm.close()，在那里也会再打印一次统计，这是正常的。

    # 整理成一个 dict 返回
    return {
        "best_score": best_score,
        "total_calls": total_calls,
        "deepseek_calls": deepseek_calls,
        "qwen_calls": qwen_calls,
    }


def main():
    all_results = []

    for run_id in range(1, N_RUNS + 1):
        stats = run_one_experiment(run_id)
        all_results.append(stats)

    # ===== 聚合统计 =====
    # 1) 平均 best_score
    valid_scores = [r["best_score"] for r in all_results if r["best_score"] is not None]
    if valid_scores:
        avg_best_score = statistics.mean(valid_scores)
    else:
        avg_best_score = None

    # 2) 聚合调用次数，算整体上的调用比例
    total_deepseek = sum(r["deepseek_calls"] for r in all_results)
    total_qwen = sum(r["qwen_calls"] for r in all_results)
    total_calls = total_deepseek + total_qwen

    if total_calls > 0:
        avg_deepseek_ratio = total_deepseek / total_calls
        avg_qwen_ratio = total_qwen / total_calls
    else:
        avg_deepseek_ratio = 0.0
        avg_qwen_ratio = 0.0

    print("\n========== Summary over all runs ==========")
    print(f"Number of runs: {N_RUNS}")
    print(f"Average best_score: {avg_best_score}")
    print(f"Total calls: {total_calls} (deepseek={total_deepseek}, qwen={total_qwen})")
    print(
        "Average call ratio (aggregated over all runs): "
        f"deepseek={avg_deepseek_ratio:.3f}, qwen={avg_qwen_ratio:.3f}"
    )


if __name__ == "__main__":
    main()
