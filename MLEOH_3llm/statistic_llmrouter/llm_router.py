from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# 你项目里已有 HttpsApi
from llm4ad.tools.llm.llm_api_https import HttpsApi


# -----------------------------
# Decision object (thread-local)
# -----------------------------
@dataclass
class RouterDecision:
    step: int
    operator: str
    prompt_sha256: str
    chosen_backend: str        # "deepseek" / "qwen" / "doubao"
    confidence: float
    reason: str


# -----------------------------
# Stats (online learning)
# -----------------------------
@dataclass
class Bucket:
    count: int = 0
    sum_reward: float = 0.0
    sum_score: float = 0.0
    fail_count: int = 0

    def mean_reward(self) -> float:
        return self.sum_reward / self.count if self.count > 0 else 0.0

    def mean_score(self) -> float:
        return self.sum_score / self.count if self.count > 0 else 0.0


class RouterStats:
    """
    stats[operator][backend] = Bucket
    operator: E1/E2/M1/M2/UNKNOWN
    backend: deepseek/qwen/doubao
    """
    OPS = ["E1", "E2", "M1", "M2", "UNKNOWN"]
    BACKENDS = ["deepseek", "qwen", "doubao"]

    def __init__(self):
        self.stats: Dict[str, Dict[str, Bucket]] = {
            op: {b: Bucket() for b in self.BACKENDS} for op in self.OPS
        }

    def update(self, operator: str, backend: str, reward: float, score: Optional[float], ok: bool):
        op = operator if operator in self.stats else "UNKNOWN"
        b = backend if backend in self.stats[op] else "deepseek"
        bucket = self.stats[op][b]
        bucket.count += 1
        bucket.sum_reward += float(reward)
        if score is not None:
            bucket.sum_score += float(score)
        if not ok:
            bucket.fail_count += 1

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for op, m in self.stats.items():
            out[op] = {}
            for b, bucket in m.items():
                out[op][b] = dataclasses.asdict(bucket)
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RouterStats":
        rs = RouterStats()
        for op, m in d.items():
            if op not in rs.stats:
                continue
            for b, bd in m.items():
                if b not in rs.stats[op]:
                    continue
                rs.stats[op][b] = Bucket(
                    count=int(bd.get("count", 0)),
                    sum_reward=float(bd.get("sum_reward", 0.0)),
                    sum_score=float(bd.get("sum_score", 0.0)),
                    fail_count=int(bd.get("fail_count", 0)),
                )
        return rs

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "RouterStats":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return RouterStats.from_dict(d)

    def summary_for_prompt(self, top_k_ops: int = 5) -> str:
        """
        压缩成 Router Prompt 里可读的一段文本：
        - 对每个 operator 给出每个 backend 的 (count, mean_reward, fail)
        """
        lines = []
        for op in self.OPS[:top_k_ops]:
            parts = []
            for b in self.BACKENDS:
                bk = self.stats[op][b]
                parts.append(f"{b}: n={bk.count}, meanR={bk.mean_reward():.3f}, fail={bk.fail_count}")
            lines.append(f"{op}: " + " | ".join(parts))
        return "\n".join(lines)


# -----------------------------
# Tracer
# -----------------------------
class LLMRouterTracer:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_file()

    def _init_file(self):
        if os.path.exists(self.path):
            return
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["step", "operator", "chosen_backend", "confidence", "reason", "prompt_sha256", "time"],
            )
            w.writeheader()

    def log(self, decision: RouterDecision):
        row = {
            "step": decision.step,
            "operator": decision.operator,
            "chosen_backend": decision.chosen_backend,
            "confidence": f"{decision.confidence:.3f}",
            "reason": decision.reason[:200].replace("\n", " "),
            "prompt_sha256": decision.prompt_sha256,
            "time": f"{time.time():.3f}",
        }
        with self._lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["step", "operator", "chosen_backend", "confidence", "reason", "prompt_sha256", "time"],
                )
                w.writerow(row)


# -----------------------------
# Feature extraction from prompt
# -----------------------------
_OP_RE = re.compile(r"(operator|op)\s*[:=]\s*(E1|E2|M1|M2)\b", flags=re.IGNORECASE)
_OP_ANY_RE = re.compile(r"\b(E1|E2|M1|M2)\b")

_ERROR_KW = [
    "traceback", "exception", "syntaxerror", "typeerror", "nameerror",
    "valueerror", "indexerror", "keyerror", "assertionerror", "indentationerror",
]


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = []
        for m in prompt:
            if isinstance(m, dict) and "content" in m:
                parts.append(str(m["content"]))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    if isinstance(prompt, dict):
        return str(prompt.get("content", prompt))
    return str(prompt)


def _detect_operator(text: str) -> str:
    m = _OP_RE.search(text)
    if m:
        return m.group(2).upper()
    m2 = _OP_ANY_RE.search(text)
    if m2:
        return m2.group(1).upper()
    return "UNKNOWN"


def _truncate_prompt(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return head + "\n...\n" + tail


# -----------------------------
# Router Prompt + parsing
# -----------------------------
def _extract_json_obj(s: str) -> Optional[str]:
    # 允许模型输出 ```json ... ``` 或前后有解释
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start:end+1]


def _safe_parse_router_output(s: str) -> Optional[Dict[str, Any]]:
    js = _extract_json_obj(s)
    if js is None:
        return None
    try:
        return json.loads(js)
    except Exception:
        return None


def _fallback_rule(operator: str, prompt_text: str) -> Tuple[str, float, str]:
    tl = prompt_text.lower()
    if any(k in tl for k in _ERROR_KW):
        return "deepseek", 0.2, "fallback: has_error"
    if operator in ("M1", "M2"):
        return "deepseek", 0.2, f"fallback: operator_{operator}"
    if operator == "E2":
        return "qwen", 0.2, "fallback: operator_E2"
    if operator == "E1":
        return "doubao", 0.2, "fallback: operator_E1"
    # 默认：便宜探索
    return "doubao", 0.2, "fallback: default"


def build_router_prompt(
    operator: str,
    prompt_text: str,
    stats_summary: str,
    backend_cards: str,
) -> str:
    return f"""You are a routing controller. Your job: choose ONE backend LLM for the next code-generation call.

Context:
- Problem: heuristic program evolution (EoH) for Online Bin Packing.
- Operator: {operator}
- The backend LLM will be asked to write/modify Python code. Choose the backend most likely to produce a correct, high-quality improvement.

Available backends:
{backend_cards}

Historical routing stats (higher meanR is better):
{stats_summary}

User prompt (truncated):
\"\"\"{prompt_text}\"\"\"

Output STRICT JSON only (no markdown, no extra text):
{{
  "backend": "deepseek" | "qwen" | "doubao",
  "confidence": 0.0-1.0,
  "reason": "short reason"
}}
"""


# -----------------------------
# LLM Router
# -----------------------------
class LLMRouterLLM:
    """
    router_llm: 用来做路由决策的 LLM（建议用便宜的，如 doubao mini）
    backends: deepseek/qwen/doubao 三个真正干活的模型
    stats: 训练阶段持续更新，测试阶段加载并冻结
    """
    def __init__(
        self,
        router_llm: HttpsApi,
        deepseek_llm: HttpsApi,
        qwen_llm: HttpsApi,
        doubao_llm: HttpsApi,
        stats: Optional[RouterStats] = None,
        freeze_stats: bool = True,
        tracer: Optional[LLMRouterTracer] = None,
        max_prompt_chars: int = 6000,
        max_router_retries: int = 1,
        verbose: bool = False,
    ):
        self.router_llm = router_llm
        self.deepseek_llm = deepseek_llm
        self.qwen_llm = qwen_llm
        self.doubao_llm = doubao_llm

        self.stats = stats or RouterStats()
        self.freeze_stats = bool(freeze_stats)
        self.tracer = tracer
        self.max_prompt_chars = int(max_prompt_chars)
        self.max_router_retries = int(max_router_retries)
        self.verbose = verbose

        self._tl = threading.local()
        self._step = 0

        # 你可以在论文里写“能力卡/价格倾向”
        self.backend_cards = (
            "- deepseek: strong debugging & code fixes; good for local edits.\n"
            "- qwen: strong restructuring & long-context; good for large rewrites.\n"
            "- doubao: low-cost fast exploration; good for generating variants.\n"
        )

    def pop_last_decision(self) -> Optional[RouterDecision]:
        d = getattr(self._tl, "last_decision", None)
        if hasattr(self._tl, "last_decision"):
            delattr(self._tl, "last_decision")
        return d

    def _set_last_decision(self, d: RouterDecision):
        self._tl.last_decision = d

    def _pick_backend_obj(self, name: str) -> HttpsApi:
        if name == "deepseek":
            return self.deepseek_llm
        if name == "qwen":
            return self.qwen_llm
        return self.doubao_llm

    def _route(self, operator: str, prompt_text: str) -> Tuple[str, float, str]:
        stats_summary = self.stats.summary_for_prompt()
        truncated = _truncate_prompt(prompt_text, self.max_prompt_chars)

        router_prompt = build_router_prompt(
            operator=operator,
            prompt_text=truncated,
            stats_summary=stats_summary,
            backend_cards=self.backend_cards,
        )

        last_err = None
        for _ in range(self.max_router_retries + 1):
            out = self.router_llm.draw_sample(router_prompt)
            obj = _safe_parse_router_output(out)
            if obj and isinstance(obj, dict):
                backend = str(obj.get("backend", "")).strip().lower()
                conf = obj.get("confidence", 0.0)
                reason = str(obj.get("reason", "")).strip()
                if backend in ("deepseek", "qwen", "doubao"):
                    try:
                        conf_f = float(conf)
                    except Exception:
                        conf_f = 0.5
                    conf_f = max(0.0, min(1.0, conf_f))
                    return backend, conf_f, reason[:300]
            last_err = out

        # fallback
        return _fallback_rule(operator, prompt_text)

    def draw_sample(self, prompt: Any, *args, **kwargs) -> Any:
        self._step += 1

        prompt_text = _prompt_to_text(prompt)
        operator = _detect_operator(prompt_text)
        sha = hashlib.sha256(prompt_text.encode("utf-8", errors="ignore")).hexdigest()

        chosen, conf, reason = self._route(operator, prompt_text)
        backend = self._pick_backend_obj(chosen)

        decision = RouterDecision(
            step=self._step,
            operator=operator,
            prompt_sha256=sha,
            chosen_backend=chosen,
            confidence=conf,
            reason=reason,
        )
        self._set_last_decision(decision)
        if self.tracer is not None:
            try:
                self.tracer.log(decision)
            except Exception:
                pass

        if self.verbose:
            print(f"[LLMRouter] step={self._step} op={operator} -> {chosen} conf={conf:.2f} reason={reason}")

        return backend.draw_sample(prompt, *args, **kwargs)

    def draw_samples(self, *args, **kwargs):
        n = kwargs.pop("n", None)
        if n is not None:
            n = int(n)
            return [self.draw_sample(*args, **kwargs) for _ in range(n)]
        return self.draw_sample(*args, **kwargs)

    # 训练阶段用：在 EoH 拿到 score 后更新 stats
    def update_stats(self, decision: RouterDecision, reward: float, score: Optional[float], ok: bool):
        if self.freeze_stats:
            return
        self.stats.update(
            operator=decision.operator,
            backend=decision.chosen_backend,
            reward=float(reward),
            score=score,
            ok=ok,
        )

    def close(self):
        for b in (self.router_llm, self.deepseek_llm, self.qwen_llm, self.doubao_llm):
            if hasattr(b, "close"):
                try:
                    b.close()
                except Exception:
                    pass