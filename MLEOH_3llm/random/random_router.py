# random_router.py
from __future__ import annotations

import hashlib
import random
from typing import Any, Optional


class RandomRouter:
    """
    Random routing baseline (supports 2 or 3 backends):

    3-backend mode:
      - 每次 draw_sample 在 deepseek / qwen / doubao 中按概率随机选一个
      - 统计 call_cnt / deepseek_cnt / qwen_cnt / doubao_cnt

    2-backend mode (backward compatible):
      - 如果 doubao_llm=None，则退化为 deepseek / qwen 二选一
      - 使用 p_model_a (deepseek 概率)

    参数：
      seed: 控制路由随机性（不影响后端 LLM 的采样随机性）
      deterministic_by_prompt: True 时用 hash(seed, prompt) 做确定性伪随机，避免并行漂移
    """

    def __init__(
        self,
        deepseek_llm,
        qwen_llm,
        doubao_llm=None,
        # ---- 3-backend probs ----
        p_deepseek: Optional[float] = None,
        p_qwen: Optional[float] = None,
        p_doubao: Optional[float] = None,
        # ---- 2-backend legacy prob ----
        p_model_a: float = 0.5,
        seed: Optional[int] = None,
        deterministic_by_prompt: bool = False,
        verbose: bool = True,
    ):
        self.deepseek_llm = deepseek_llm
        self.qwen_llm = qwen_llm
        self.doubao_llm = doubao_llm

        self.seed = seed
        self.deterministic_by_prompt = deterministic_by_prompt
        self.verbose = verbose

        # RNG (only affects routing decision)
        self._rng = random.Random(seed)

        # stats
        self.call_cnt = 0
        self.deepseek_cnt = 0
        self.qwen_cnt = 0
        self.doubao_cnt = 0

        # mode & probabilities
        self.three_backend = doubao_llm is not None

        if self.three_backend:
            # defaults
            if p_deepseek is None and p_qwen is None and p_doubao is None:
                p_deepseek = 1.0 / 3.0
                p_qwen = 1.0 / 3.0
                p_doubao = 1.0 - p_deepseek - p_qwen
            else:
                if p_deepseek is None:
                    p_deepseek = 0.0
                if p_qwen is None:
                    p_qwen = 0.0
                if p_doubao is None:
                    p_doubao = 1.0 - float(p_deepseek) - float(p_qwen)

            self.p_deepseek = float(p_deepseek)
            self.p_qwen = float(p_qwen)
            self.p_doubao = float(p_doubao)

            if not (0.0 <= self.p_deepseek <= 1.0 and 0.0 <= self.p_qwen <= 1.0 and 0.0 <= self.p_doubao <= 1.0):
                raise ValueError(f"Probabilities must be in [0,1]. Got ds={self.p_deepseek}, qw={self.p_qwen}, db={self.p_doubao}")
            s = self.p_deepseek + self.p_qwen + self.p_doubao
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"Probabilities must sum to 1. Got sum={s} (ds={self.p_deepseek}, qw={self.p_qwen}, db={self.p_doubao})")

            if self.verbose:
                mode = "hash(prompt)" if deterministic_by_prompt else "rng"
                print(f"[RandomRouter] init(3-backend): ds={self.p_deepseek:.3f}, qw={self.p_qwen:.3f}, db={self.p_doubao:.3f}, seed={self.seed}, mode={mode}")
        else:
            # 2-backend legacy
            if not (0.0 <= p_model_a <= 1.0):
                raise ValueError("p_model_a must be in [0,1]")
            self.p_model_a = float(p_model_a)

            if self.verbose:
                mode = "hash(prompt)" if deterministic_by_prompt else "rng"
                print(f"[RandomRouter] init(2-backend): p_model_a={self.p_model_a:.3f}, seed={self.seed}, mode={mode}")

    def _u01_from_prompt(self, prompt_text: str) -> float:
        s = f"{self.seed}||{prompt_text}".encode("utf-8", errors="ignore")
        h = hashlib.sha256(s).digest()
        x = int.from_bytes(h[:8], "big")
        return x / 2**64

    def _sample_choice(self, prompt_text: str) -> str:
        if self.deterministic_by_prompt:
            u = self._u01_from_prompt(prompt_text)
        else:
            u = self._rng.random()

        if not self.three_backend:
            return "deepseek" if u < self.p_model_a else "qwen"

        # 3-backend categorical
        if u < self.p_deepseek:
            return "deepseek"
        if u < self.p_deepseek + self.p_qwen:
            return "qwen"
        return "doubao"

    def _choose_backend(self, prompt_text: str):
        backend_name = self._sample_choice(prompt_text)

        if backend_name == "deepseek":
            self.deepseek_cnt += 1
            backend = self.deepseek_llm
        elif backend_name == "qwen":
            self.qwen_cnt += 1
            backend = self.qwen_llm
        else:
            self.doubao_cnt += 1
            backend = self.doubao_llm

        if self.verbose:
            if self.three_backend:
                print(f"[RandomRouter] choose={backend_name} (ds={self.p_deepseek:.3f}, qw={self.p_qwen:.3f}, db={self.p_doubao:.3f})")
            else:
                print(f"[RandomRouter] choose={backend_name} (pA={self.p_model_a:.3f})")

        return backend

    def draw_sample(self, *args, **kwargs) -> Any:
        self.call_cnt += 1

        if "prompt" in kwargs:
            prompt_text = kwargs["prompt"]
        elif len(args) > 0:
            prompt_text = args[0]
        else:
            prompt_text = ""

        backend = self._choose_backend(str(prompt_text))
        return backend.draw_sample(*args, **kwargs)

    def draw_samples(self, *args, **kwargs):
        n = kwargs.pop("n", None)
        if n is not None:
            n = int(n)
            return [self.draw_sample(*args, **kwargs) for _ in range(n)]
        return self.draw_sample(*args, **kwargs)

    def close(self):
        if self.verbose:
            print("[RandomRouter] 调用统计：")
            print(f"  总 draw_sample 次数: {self.call_cnt}")
            print(f"  deepseek 被选次数  : {self.deepseek_cnt}")
            print(f"  qwen 被选次数      : {self.qwen_cnt}")
            if self.three_backend:
                print(f"  doubao 被选次数    : {self.doubao_cnt}")

        seen = set()
        backends = [self.deepseek_llm, self.qwen_llm]
        if self.three_backend:
            backends.append(self.doubao_llm)

        for backend in backends:
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