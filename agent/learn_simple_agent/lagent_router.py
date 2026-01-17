from typing import Any, Optional


# ===================== Router Prompt Template =====================

DEFAULT_DEEPSEEK_DESC = """\
- Good at understanding MULTIPLE given algorithms and synthesizing high-level patterns across them.
- Strong global reasoning: can compare different heuristic ideas and design NEW variants that mix them.
- Comfortable writing relatively long Python code blocks from scratch, including control flow and vectorized NumPy code.
- Good at exploratory / high-variance proposals that may drastically change the search behaviour of the heuristic.
"""

DEFAULT_QWEN_DESC = """\
- Very careful, conservative, and detail-oriented when editing code.
- Strong at LOCAL edits and REFINEMENT of an existing algorithm:
  * adjusting weights, coefficients, and thresholds,
  * changing parameter settings of a score function,
  * improving robustness, boundary handling, and code safety.
- Good at reading ONE given algorithm and carefully modifying parts of the logic
  while keeping the overall structure and invariants intact.
- Tends to respect types and constraints and avoid introducing obvious bugs.
"""

ROUTER_PROMPT_TEMPLATE = """\
You are a routing agent in an automatic heuristic design (AHD) system
for combinatorial optimization problems (e.g., online bin packing).

You must choose which backend LLM should handle a given prompt that
will be sent to generate or modify a heuristic program (usually Python code).

There are TWO backend models, anonymized as A and B:

[MODEL A]
{model_a_desc}

[MODEL B]
{model_b_desc}

IMPORTANT:
- Your ONLY goal is to maximize the final solution quality by choosing
  the most suitable model for THIS prompt.
- You are NOT allowed to choose based on speed, quota, or randomness.
- You must decide purely from the content of the prompt.

AVAILABLE INFORMATION:
You will see the FULL prompt that would be sent to a backend model.

DECISION FORMAT (VERY IMPORTANT):
1. First, briefly explain in 1–3 sentences which model is more suitable and why.
2. Then, on the LAST line, OUTPUT EXACTLY ONE LETTER:
      A
   or
      B

Rules:
- The LAST non-empty line must contain ONLY that one letter (A or B).
- Do NOT add any extra text after that last line.

========================================
PROMPT TO ROUTE (this is what the backend LLM would receive):
----------------------------------------
{prompt_text}
----------------------------------------

Now, decide which backend model is more suitable for this prompt.
Remember: put your explanation first, and on the LAST line output ONLY one letter (A or B).
"""



class AgentBasedRouter:
    """
    用一个 LLM 作为 agent，根据当前要发送给底层 LLM 的 prompt，
    决定这次调用用 deepseek 还是 qwen。

    - deepseek_llm / qwen_llm：真正干活生成启发式的两个底层 LLM
    - router_llm：路由 agent 用的 LLM，可以和其中一个共用（例如用 qwen，当作便宜的路由器）

    与静态版本的区别：
    - deepseek_desc / qwen_desc 可以在外部学习与更新，只覆盖能力描述部分。
    """

    def __init__(
        self,
        deepseek_llm,
        qwen_llm,
        router_llm,
        deepseek_desc: Optional[str] = None,
        qwen_desc: Optional[str] = None,
        verbose: bool = True,
    ):
        self.deepseek_llm = deepseek_llm
        self.qwen_llm = qwen_llm
        self.router_llm = router_llm  # 负责“选谁出场”的 agent
        self.verbose = verbose

        # 可学习的两段描述；如果不传，则使用默认静态描述

        self.deepseek_desc = deepseek_desc or DEFAULT_DEEPSEEK_DESC
        self.qwen_desc = qwen_desc or DEFAULT_QWEN_DESC

        if verbose:
            print("In init!!")
            print(self.deepseek_desc)
            print(self.qwen_desc)

        # 统计信息
        self.call_cnt = 0          # 总调用次数（draw_sample 次数）
        self.deepseek_cnt = 0      # 被路由到 deepseek 的次数
        self.qwen_cnt = 0          # 被路由到 qwen 的次数

    # ========= 1. 构造 router prompt 并调用 router-LLM 决策 =========
    def _build_router_prompt(self, prompt_text: str) -> str:
        return ROUTER_PROMPT_TEMPLATE.format(
            model_a_desc=self.deepseek_desc.strip(),  # A ↔ deepseek_llm
            model_b_desc=self.qwen_desc.strip(),      # B ↔ qwen_llm
            prompt_text=prompt_text,
        )

    def _ask_router_agent(self, prompt_text: str) -> str:
        """
        构造一个路由 prompt，让 router_llm 给出：
        - 1~几行的理由
        - 最后一行只输出 DEEPSEEK 或 QWEN
        """
        # print("before build prompt")
        router_prompt = self._build_router_prompt(prompt_text)
        # print("router_prompt!!!!!!!!!!",router_prompt)
        try:
            # 假设 llm4ad 的 HttpsApi 接口是 draw_sample(prompt=...)
            decision_text = self.router_llm.draw_sample(prompt=router_prompt)
        except TypeError:
            # 兼容一些旧版本：直接把字符串当第一个位置参数传入
            decision_text = self.router_llm.draw_sample(router_prompt)

        if self.verbose:
            print("========== Router-Agent decision ==========")
            print(decision_text)
            print("===========================================")

        # 解析最后一行的模型名称
        lines = [ln.strip() for ln in str(decision_text).splitlines() if ln.strip()]
        if not lines:
            if self.verbose:
                print("[Router-Agent] 空输出，默认使用 deepseek")
            return "deepseek"

        model_line = lines[-1].upper()

        # 理想情况：最后一行就是 A 或 B
        if model_line == "A":
            return "deepseek"
        if model_line == "B":
            return "qwen"

        # 如果最后一行写成 "MODEL A" / "MODEL B" 之类，做一次兜底匹配
        if "A" in model_line and "B" not in model_line:
            return "deepseek"
        if "B" in model_line and "A" not in model_line:
            return "qwen"

        # fallback：默认 A -> deepseek
        if self.verbose:
            print("[Router-Agent] 无法解析模型，默认使用模型 A (deepseek)")
        return "deepseek"

    # ========= 2. 根据 router 决策选后端 =========
    def _choose_backend(self, prompt_text: str):
        print("in choose backend")
        backend_name = self._ask_router_agent(prompt_text)

        if backend_name == "qwen":
            backend = self.qwen_llm
            self.qwen_cnt += 1
        else:
            backend_name = "deepseek"
            backend = self.deepseek_llm
            self.deepseek_cnt += 1

        
        print(f"[Router] 使用模型: {backend_name}\n")

        return backend

    # ========= 3. EoH 会调用的接口 =========
    def draw_sample(self, *args, **kwargs) -> Any:
        """
        这里不再自己写 if-else 判断 prompt 类型，
        而是把 prompt 交给 router-LLM 选后端。
        """

        self.call_cnt += 1  # 总调用次数 +1

        # 尽量从参数中拿出“要发到底层 LLM 的 prompt 文本”，仅用于路由决策
        if "prompt" in kwargs:
            prompt_text = kwargs["prompt"]
        elif len(args) > 0:
            prompt_text = args[0]
        else:
            prompt_text = ""

        backend = self._choose_backend(str(prompt_text))

        # 把所有参数原封不动转发给被选中的底层 LLM
        return backend.draw_sample(*args, **kwargs)

    def draw_samples(self, *args, **kwargs):
        """
        如果 llm4ad 调用的是 draw_samples（不一定会用到），
        这里简单地循环调用 draw_sample。
        """
        n = kwargs.pop("n", None)
        if n is not None:
            results = []
            for _ in range(n):
                results.append(self.draw_sample(*args, **kwargs))
            return results

        return self.draw_sample(*args, **kwargs)

    # ========= 4. 让 EoH.run() 能正常关闭资源 =========
    def close(self):
        """
        EoH 在结束时会调用 llm.close()，这里要实现一下。
        把三个底层 client 都尝试关掉（去重避免重复 close）。
        同时打印统计信息。
        """

        if self.verbose:
            print("[Router] 调用统计：")
            print(f"  总 draw_sample 次数: {self.call_cnt}")
            print(f"  deepseek 被选次数: {self.deepseek_cnt}")
            print(f"  qwen 被选次数    : {self.qwen_cnt}")

        seen = set()
        for backend in (self.deepseek_llm, self.qwen_llm, self.router_llm):
            if backend is None:
                continue
            if id(backend) in seen:
                continue
            seen.add(id(backend))
            if hasattr(backend, "close"):
                try:
                    backend.close()
                except Exception as e:
                    if self.verbose:
                        print(f"[Router] 关闭 backend 时出错: {e}")
