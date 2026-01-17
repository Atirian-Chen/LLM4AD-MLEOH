# fixed_prompt_router.py
from typing import Any

class FixedPromptRouter:
    """
    一个最简单的“静态 prompt router”：
    - 外部接口：只要有 draw_sample / draw_samples，就可以当作 llm 给 EoH 用
    - 内部：持有 deepseek_llm 和 qwen_llm 两个 HttpsApi 实例
    - route() 里的规则是“写死的”，你可以根据需要修改
    """

    def __init__(self, deepseek_llm, qwen_llm, verbose: bool = True):
        self.deepseek_llm = deepseek_llm
        self.qwen_llm = qwen_llm
        self.verbose = verbose

    # ===== 路由策略：根据 prompt 决定用哪个模型 =====
    def _choose_backend(self, prompt_text: str):
        """
        这里你可以根据自己的想法写规则：
        - 比如 E1 / E2 用 DeepSeek，M1/M2/M3 用 Qwen
        - 或者根据关键字、prompt 长度、是否写 code 等来分
        """

        text = (prompt_text or "").lower()

        # 例子 1：如果 prompt 明确是“探索类”（你可以根据 EoH 的模板关键字来改）
        # 假设 E1/E2 的 prompt 里会出现 “exploration” 或 “search for new heuristics”
        if "e1" in text or "e2" in text or "exploration" in text:
            backend_name = "deepseek"
            backend = self.deepseek_llm

        # 例子 2：如果 prompt 属于“mutation/修改类”，就让 Qwen 来做
        elif "m1" in text or "m2" in text or "mutation" in text:
            backend_name = "qwen"
            backend = self.qwen_llm

        # 例子 3：如果 prompt 里出现“refine” 或 “improve code”，也给 Qwen
        elif "refine" in text or "improve the heuristic" in text:
            backend_name = "qwen"
            backend = self.qwen_llm

        # 默认策略：谁便宜就给谁（这里假设你更想用 DeepSeek，多数情况就走 DeepSeek）
        else:
            backend_name = "deepseek"
            backend = self.deepseek_llm

        if self.verbose:
            print(f"[Router] 使用模型: {backend_name}")

        return backend

    # ===== 核心接口：EoH 会调用这个 =====
    def draw_sample(self, *args, **kwargs) -> Any:
        """
        为了兼容 llm4ad 的 sampler 接口，这里不假设具体签名，
        而是从 args/kwargs 里尽量拿到 prompt，再把所有参数原封不动转发。
        """

        # 1) 尝试从 kwargs 里找 prompt
        prompt_text = None
        if "prompt" in kwargs:
            prompt_text = kwargs["prompt"]
        elif len(args) > 0:
            # 很多实现都是第一个位置参数就是 prompt/text
            prompt_text = args[0]

        # 2) 基于 prompt 做路由
        backend = self._choose_backend(prompt_text if isinstance(prompt_text, str) else str(prompt_text))

        # 3) 把调用转发给对应的 HttpsApi 实例
        return backend.draw_sample(*args, **kwargs)

    # ===== 有些方法可能会调用 draw_samples，这里给一个简单实现 =====
    def draw_samples(self, *args, **kwargs):
        """
        如果 llm4ad 调用 draw_samples，我们就简单地循环调用 draw_sample。
        具体是否会调用，要看版本；写上更保险。
        """

        n = kwargs.pop("n", None)

        # 如果使用 kwargs 里的 n 来控制采样数量
        if n is not None:
            results = []
            for _ in range(n):
                results.append(self.draw_sample(*args, **kwargs))
            return results

        # 如果没有 n，可能是其他自定义签名，就直接调用一次
        return self.draw_sample(*args, **kwargs)
