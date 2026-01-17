# test_eoh_obp_router.py
import os

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

from agent_router import AgentBasedRouter


def main():
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
        model="qwen-flash",   # 或你之前用的 qwen 模型
        timeout=60,
    )

    # 路由 agent：为了省钱，可以直接复用 Qwen 当 router 模型
    router_agent_llm = qwen_llm      # 如果你想让 DeepSeek 当 router，也可以改成 deepseek_llm

    # 把三个拼成一个逻辑上的 “llm”
    router_llm = AgentBasedRouter(
        deepseek_llm=deepseek_llm,
        qwen_llm=qwen_llm,
        router_llm=router_agent_llm,
        verbose=True,
    )

    # ===== 2. 任务和 profiler =====
    task = OBPEvaluation()

    log_dir = os.path.join("logs", "eoh_obp_router_agent")
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

        num_samplers=4,          # 建议先设 1，确认没问题再开多进程
        num_evaluators=4,

        debug_mode=False,
    )

    # ===== 4. 运行 =====
    print(">>> Start EoH (Agent Router + DeepSeek + Qwen) on Online Bin Packing...")
    best_program = method.run()
    print(">>> Search finished.")
    print(">>> Best score:\n")
    print(method.best_score)


if __name__ == "__main__":
    main()
