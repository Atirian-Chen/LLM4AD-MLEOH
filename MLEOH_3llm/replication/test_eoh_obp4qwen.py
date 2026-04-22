import os
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler


def main():
    # ===== 1. 配置 LLM =====
    # 把这里换成你自己的 OpenAI API Key
    API_KEY = "sk-2af91af9547d470493627d0a0d95edfa"

    # 如果你想尽量贴近 EoH 原论文，可以用 gpt-3.5-turbo
    # 如果没有，就用你有权限的任意 chat/completions 模型
    llm = HttpsApi(
        host="dashscope.aliyuncs.com",       # 注意：不要写成 https://api.openai.com
        key=API_KEY,
        model="qwen-flash",       # 或 gpt-4o-mini, gpt-4.1 等
        timeout=60,
    )

    # ===== 2. 选择任务：Online Bin Packing =====
    # 这是 LLM4AD 自带的 OBP 评估接口
    task = OBPEvaluation()

    # ===== 3. 配置日志与输出路径 =====
    log_dir = os.path.join("logs", "eoh_obp_cli_qwen")
    os.makedirs(log_dir, exist_ok=True)

    profiler = EoHProfiler(
        log_dir=log_dir,
        log_style="simple",   # 或 "tensorboard" / "wandb"，先用 simple 最省事
    )

    # ===== 4. 配置 EoH 超参数 =====
    # 下面这组参数是“比较像论文里设置”的版本，可以根据机器和预算再调
    method = EoH(
        llm=llm,
        evaluation=task,
        profiler=profiler,

        # 搜索规模相关
        pop_size=10,          # 种群大小
        max_generations=6,   # 最大迭代代数
        max_sample_nums=10000, # LLM 调用总预算（近似）

        selection_num=3,      # 每轮从种群中选多少个父代供 E1/E2 使用

        # 启用哪些算子（EoH 原论文里都是开的）
        use_e2_operator=True,
        use_m1_operator=True,
        use_m2_operator=True,

        # 并行度（看你机器 CPU 核心数，先设 2/4 都行）
        num_samplers=4,
        num_evaluators=4,

        debug_mode=False,
    )

    # ===== 5. 运行 EoH 搜索 =====
    print(">>> Start EoH on Online Bin Packing...")
    best_program = method.run()
    print(">>> Search finished.")
    # print(">>> Best score:", best_score)
    print(">>> Best program:\n")
    print(best_program)


if __name__ == "__main__":
    main()
