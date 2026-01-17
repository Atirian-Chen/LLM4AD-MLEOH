# test_eoh_obp_router.py
import os

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

from simple_agent.fixed_prompt_router import FixedPromptRouter


def main():
    # ===== 1. 配置两个底层 LLM =====
    # ⚠️ 把 key 换成你自己的，别用我随便写的
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

    # 1.3 用固定 Router 把两个合并成一个“逻辑上的 LLM”
    router_llm = FixedPromptRouter(
        deepseek_llm=deepseek_llm,
        qwen_llm=qwen_llm,
        verbose=True,      # 想关掉路由打印就设成 False
    )

    # ===== 2. 任务：Online Bin Packing =====
    task = OBPEvaluation()

    # ===== 3. 日志路径 =====
    log_dir = os.path.join("logs", "eoh_obp_router_ds_qw")
    os.makedirs(log_dir, exist_ok=True)

    profiler = EoHProfiler(
        log_dir=log_dir,
        log_style="simple",
    )

    # ===== 4. 配置 EoH 超参数 =====
    # ⚠️ 第一次为了避免多进程 + Router 搞出奇怪的 pickling 问题，
    #    建议先把 num_samplers=1 跑通，再开到 4。
    method = EoH(
        llm=router_llm,      # ✅ 这里用 router，而不是单一 HttpsApi
        evaluation=task,
        profiler=profiler,

        pop_size=10,
        max_generations=10,
        max_sample_nums=20000,

        selection_num=4,
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,

        num_samplers=4,      # 先设 1 保守一点，跑通之后再改回 4
        num_evaluators=4,

        debug_mode=False,
    )

    # ===== 5. 运行 =====
    print(">>> Start EoH (Router + DeepSeek + Qwen) on Online Bin Packing...")
    best_program = method.run()
    print(">>> Search finished.")
    print(">>> Best program:\n")
    print(best_program)


if __name__ == "__main__":
    main()
