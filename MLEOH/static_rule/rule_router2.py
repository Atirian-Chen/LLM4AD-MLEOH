# rule_router.py
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


class RuleRouter:
    """
    Rule-based routing baseline (no router LLM):
    - 从 prompt_text 提取特征（报错、operator(E1/E2/M1/M2)、长度、代码密度、关键词）
    - 用固定优先级规则选择 backend A/B
    - 接口与 AgentBasedRouter / RandomRouter 对齐：draw_sample / draw_samples / close
    - 统计 call_cnt/deepseek_cnt/qwen_cnt，并记录 rule 命中分布，方便写报告
    """

    ERROR_KEYWORDS = [
        "traceback",
        "exception",
        "syntaxerror",
        "typeerror",
        "nameerror",
        "valueerror",
        "indexerror",
        "keyerror",
        "assertionerror",
        "indentationerror",
        "unboundlocalerror",
        "zerodivisionerror",
    ]

    # 一些弱信号关键词（不如 ERROR 强，但对 default 分支有帮助）
    FIX_WORDS = ["fix", "bug", "repair", "patch", "debug", "crash", "error", "exception"]
    EDIT_WORDS = ["modify", "edit", "refine", "tweak", "adjust", "improve", "optimize"]
    EXPLORE_WORDS = ["design", "propose", "invent", "rewrite", "new heuristic", "explore", "diversify"]

    def __init__(
        self,
        deepseek_llm,
        qwen_llm,
        # 阈值（可在脚本里调）
        long_prompt_chars: int = 12000,
        long_prompt_lines: int = 260,
        dense_code_markers: int = 18,
        # 默认落点
        default_backend: str = "qwen",  # or "qwen"
        verbose: bool = False,
    ):
        self.deepseek_llm = deepseek_llm
        self.qwen_llm = qwen_llm

        self.long_prompt_chars = int(long_prompt_chars)
        self.long_prompt_lines = int(long_prompt_lines)
        self.dense_code_markers = int(dense_code_markers)
        self.default_backend = default_backend
        self.verbose = verbose

        # 统计字段（与现有 router 保持一致）
        self.call_cnt = 0
        self.deepseek_cnt = 0
        self.qwen_cnt = 0

        # 额外统计（写进 result.json）
        self.rule_hits: Dict[str, int] = {}
        self.last_decision: Optional[Dict[str, Any]] = None

        if self.verbose:
            print(
                f"[RuleRouter] init: default={self.default_backend}, "
                f"long_chars={self.long_prompt_chars}, long_lines={self.long_prompt_lines}, "
                f"dense_code_markers={self.dense_code_markers}"
            )

    # --------------------
    # Feature extraction
    # --------------------
    def _detect_operator(self, text: str) -> Optional[str]:
        """
        尝试从 prompt 中识别 E1/E2/M1/M2。
        兼容 'Operator: E1' / 'E1' / 'op=E2' 等情况。
        """
        # 强信号：含 Operator 或 op 标识
        m = re.search(r"(operator|op)\s*[:=]\s*(E1|E2|M1|M2)\b", text, flags=re.IGNORECASE)
        if m:
            return m.group(2).upper()

        # 弱信号：只出现 E1/E2/M1/M2 也算（可能有误判，但做基线够用）
        m2 = re.search(r"\b(E1|E2|M1|M2)\b", text)
        if m2:
            return m2.group(1).upper()

        return None

    def _extract_features(self, prompt_text: str) -> Dict[str, Any]:
        t = prompt_text or ""
        tl = t.lower()

        lines = t.splitlines()
        n_lines = len(lines)
        n_chars = len(t)

        # 简易代码密度估计：出现 def/class/```/return/for/while 的次数
        code_markers = 0
        for pat in ["def ", "class ", "```", "return ", " for ", " while ", "import ", "np.", "numpy"]:
            code_markers += tl.count(pat)

        has_error = any(k in tl for k in self.ERROR_KEYWORDS)
        operator = self._detect_operator(t)

        # 关键词信号（弱）
        fix_score = sum(1 for w in self.FIX_WORDS if w in tl)
        edit_score = sum(1 for w in self.EDIT_WORDS if w in tl)
        explore_score = sum(1 for w in self.EXPLORE_WORDS if w in tl)

        is_long = (n_chars >= self.long_prompt_chars) or (n_lines >= self.long_prompt_lines)
        is_code_dense = code_markers >= self.dense_code_markers

        return {
            "n_chars": n_chars,
            "n_lines": n_lines,
            "code_markers": code_markers,
            "has_error": has_error,
            "operator": operator,
            "fix_score": fix_score,
            "edit_score": edit_score,
            "explore_score": explore_score,
            "is_long": is_long,
            "is_code_dense": is_code_dense,
        }

    # --------------------
    # Rule policy
    # --------------------
    def _hit(self, rule_name: str) -> None:
        self.rule_hits[rule_name] = self.rule_hits.get(rule_name, 0) + 1

    def _choose_backend(self, prompt_text: str) -> Tuple[str, Any, str, Dict[str, Any]]:
        feats = self._extract_features(prompt_text)

        backend_name: str
        reason: str

        # Rule priority (高优先级在前)

        # R1: 明显报错 / 修复
        if feats["has_error"]:
            backend_name = "deepseek"
            reason = "has_error_traceback_or_exception"
            self._hit("R1_error_fix")

        # R2: M1/M2 - 局部编辑/微调
        elif feats["operator"] in ("M1", "M2"):
            backend_name = "deepseek"
            reason = f"operator_{feats['operator']}_local_edit"
            self._hit("R2_mutation_edit")

        # R3: E2 - 大改/结构重构
        elif feats["operator"] == "E2":
            backend_name = "qwen"
            reason = "operator_E2_global_restructure"
            self._hit("R3_expand_restructure")

        # R4: E1 - 探索变体
        elif feats["operator"] == "E1":
            backend_name = "qwen"
            reason = "operator_E1_exploration"
            self._hit("R4_expand_variant")

        # R5: prompt 太长/代码密度太高（按你的假设可调整）
        elif feats["is_long"] or feats["is_code_dense"]:
            backend_name = "qwen"
            reason = "long_or_code_dense_global_context"
            self._hit("R5_long_or_dense")

        # R6: 弱信号：更像编辑/修补
        elif feats["fix_score"] + feats["edit_score"] >= feats["explore_score"] + 2:
            backend_name = "deepseek"
            reason = "keyword_bias_toward_edit_fix"
            self._hit("R6_keywords_edit")

        # Default
        else:
            backend_name = self.default_backend
            reason = "default"
            self._hit("R0_default")

        if backend_name == "qwen":
            backend = self.qwen_llm
            self.qwen_cnt += 1
        else:
            backend_name = "deepseek"
            backend = self.deepseek_llm
            self.deepseek_cnt += 1

        decision = {
            "backend": backend_name,
            "reason": reason,
            "features": feats,
        }
        self.last_decision = decision

        if self.verbose:
            print(f"[RuleRouter] choose={backend_name} reason={reason} feats={feats}")

        return backend_name, backend, reason, feats

    # --------------------
    # llm4ad interface
    # --------------------
    def draw_sample(self, *args, **kwargs) -> Any:
        self.call_cnt += 1

        if "prompt" in kwargs:
            prompt_text = kwargs["prompt"]
        elif len(args) > 0:
            prompt_text = args[0]
        else:
            prompt_text = ""

        _, backend, _, _ = self._choose_backend(str(prompt_text))
        return backend.draw_sample(*args, **kwargs)

    def draw_samples(self, *args, **kwargs):
        n = kwargs.pop("n", None)
        if n is not None:
            n = int(n)
            return [self.draw_sample(*args, **kwargs) for _ in range(n)]
        return self.draw_sample(*args, **kwargs)

    def close(self):
        if self.verbose:
            print("[RuleRouter] 调用统计：")
            print(f"  总 draw_sample 次数: {self.call_cnt}")
            print(f"  deepseek 被选次数: {self.deepseek_cnt}")
            print(f"  qwen 被选次数    : {self.qwen_cnt}")
            print(f"  规则命中分布      : {self.rule_hits}")

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
                        print(f"[RuleRouter] 关闭 backend 时出错: {e}")
