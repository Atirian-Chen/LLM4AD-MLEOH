# random_router.py
from __future__ import annotations

import hashlib
import random
from typing import Any, Optional


class RandomRouter:
    """
    Random routing baseline:
    - 每次 draw_sample 随机选择 deepseek_llm 或 qwen_llm
    - 实现与 AgentBasedRouter 相同的接口：draw_sample / draw_samples / close
    - 统计 call_cnt / deepseek_cnt / qwen_cnt，便于对比

    参数：
      p_model_a: 选择模型A（这里对应 deepseek_llm）的概率，默认0.5
      seed: 控制路由随机性（与后端模型本身的采样随机性无关）
      deterministic_by_prompt: True 时用 hash(seed, prompt) 做“确定性伪随机”，
                               可以避免并行导致的随机序列漂移
    """

    def __init__(
        self,
        deepseek_llm,
        qwen_llm,
        p_model_a: float = 0.5,
        seed: Optional[int] = None,
        deterministic_by_prompt: bool = False,
        verbose: bool = True,
    ):
        assert 0.0 <= p_model_a <= 1.0, "p_model_a must be in [0, 1]"
        self.deepseek_llm = deepseek_llm
        self.qwen_llm = qwen_llm

        self.p_model_a = float(p_model_a)
        self.seed = seed
        self.deterministic_by_prompt = deterministic_by_prompt
        self.verbose = verbose

        # 随机数发生器（只影响“路由选择”，不影响后端 LLM 的采样）
        self._rng = random.Random(seed)

        # 统计信息（保持与你现有 router 一样的字段名，便于复用分析脚本）
        self.call_cnt = 0
        self.deepseek_cnt = 0
        self.qwen_cnt = 0

        if self.verbose:
            mode = "hash(prompt)" if deterministic_by_prompt else "rng"
            print(f"[RandomRouter] init: p_model_a={self.p_model_a}, seed={self.seed}, mode={mode}")

    def _u01_from_prompt(self, prompt_text: str) -> float:
        """
        用 (seed, prompt_text) 生成一个 [0,1) 的伪随机数。
        这样在并行条件下也可复现：同一个 prompt 永远路由到同一个模型（给定 seed）。
        """
        s = f"{self.seed}||{prompt_text}".encode("utf-8", errors="ignore")
        h = hashlib.sha256(s).digest()
        x = int.from_bytes(h[:8], "big")  # 64-bit
        return x / 2**64

    def _sample_choice(self, prompt_text: str) -> str:
        """
        返回 "deepseek" 或 "qwen"
        """
        if self.deterministic_by_prompt:
            u = self._u01_from_prompt(prompt_text)
        else:
            u = self._rng.random()

        return "deepseek" if u < self.p_model_a else "qwen"

    def _choose_backend(self, prompt_text: str):
        backend_name = self._sample_choice(prompt_text)

        if backend_name == "deepseek":
            self.deepseek_cnt += 1
            backend = self.deepseek_llm
        else:
            self.qwen_cnt += 1
            backend = self.qwen_llm

        if self.verbose:
            print(f"[RandomRouter] choose={backend_name} (pA={self.p_model_a})")

        return backend

    def draw_sample(self, *args, **kwargs) -> Any:
        """
        EoH 会调用的接口：与 AgentBasedRouter 一致
        """
        self.call_cnt += 1

        # 抽取 prompt 文本（仅用于路由，不修改内容）
        if "prompt" in kwargs:
            prompt_text = kwargs["prompt"]
        elif len(args) > 0:
            prompt_text = args[0]
        else:
            prompt_text = ""

        backend = self._choose_backend(str(prompt_text))
        return backend.draw_sample(*args, **kwargs)

    def draw_samples(self, *args, **kwargs):
        """
        兼容 llm4ad 的 draw_samples(n=...) 调用方式
        """
        n = kwargs.pop("n", None)
        if n is not None:
            n = int(n)
            return [self.draw_sample(*args, **kwargs) for _ in range(n)]
        return self.draw_sample(*args, **kwargs)

    def close(self):
        """
        与 AgentBasedRouter 一致：打印统计并关闭后端
        """
        if self.verbose:
            print("[RandomRouter] 调用统计：")
            print(f"  总 draw_sample 次数: {self.call_cnt}")
            print(f"  deepseek 被选次数: {self.deepseek_cnt}")
            print(f"  qwen 被选次数    : {self.qwen_cnt}")

        seen = set()
        for backend in (self.deepseek_llm, self.qwen_llm):
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
                        print(f"[RandomRouter] 关闭 backend 时出错: {e}")
