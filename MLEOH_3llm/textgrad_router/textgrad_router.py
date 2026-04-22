from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llm4ad.tools.llm.llm_api_https import HttpsApi


# =========================
# Helpers
# =========================
_OP_RE = re.compile(r"(operator|op)\s*[:=]\s*(E1|E2|M1|M2)\b", flags=re.IGNORECASE)
_OP_ANY_RE = re.compile(r"\b(E1|E2|M1|M2)\b")
_ERROR_KW = [
    "traceback", "exception", "syntaxerror", "typeerror", "nameerror",
    "valueerror", "indexerror", "keyerror", "assertionerror", "indentationerror",
]


def prompt_to_text(prompt: Any) -> str:
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


def detect_operator(text: str) -> str:
    m = _OP_RE.search(text)
    if m:
        return m.group(2).upper()
    m2 = _OP_ANY_RE.search(text)
    if m2:
        return m2.group(1).upper()
    return "UNKNOWN"


def truncate(text: str, max_chars: int = 5000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return head + "\n...\n" + tail


def extract_json_obj(s: str) -> Optional[str]:
    if not s:
        return None
    a = s.find("{")
    b = s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    return s[a:b + 1]


def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    js = extract_json_obj(s)
    if js is None:
        return None
    try:
        return json.loads(js)
    except Exception:
        return None


# =========================
# Data structures
# =========================
@dataclass
class TGDecision:
    step: int
    operator: str
    prompt_sha256: str
    backend: str
    confidence: float
    reason: str
    policy_version: int


@dataclass
class TGOutcome:
    step: int
    operator: str
    backend: str
    ok: bool
    score: Optional[float]
    reward: float
    policy_version: int


# =========================
# Tracing
# =========================
class TGTracer:
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
                fieldnames=[
                    "step", "operator", "backend", "confidence", "reason",
                    "ok", "score", "reward", "policy_version", "prompt_sha256", "time"
                ],
            )
            w.writeheader()

    def log(self, decision: TGDecision, outcome: Optional[TGOutcome]):
        row = {
            "step": decision.step,
            "operator": decision.operator,
            "backend": decision.backend,
            "confidence": f"{decision.confidence:.3f}",
            "reason": decision.reason[:200].replace("\n", " "),
            "ok": "" if outcome is None else str(bool(outcome.ok)),
            "score": "" if (outcome is None or outcome.score is None) else f"{float(outcome.score):.6f}",
            "reward": "" if outcome is None else f"{float(outcome.reward):.6f}",
            "policy_version": decision.policy_version,
            "prompt_sha256": decision.prompt_sha256,
            "time": f"{time.time():.3f}",
        }
        with self._lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step", "operator", "backend", "confidence", "reason",
                        "ok", "score", "reward", "policy_version", "prompt_sha256", "time"
                    ],
                )
                w.writerow(row)


# =========================
# Policy state + TextGrad updates
# =========================
DEFAULT_POLICY = """[Routing Policy v0]
- If prompt contains traceback/exception or explicit error: choose deepseek.
- If operator is M1 or M2 (local mutation/edit): choose deepseek.
- If operator is E2 (global restructure): choose qwen.
- If operator is E1 (exploration/variant generation): choose doubao.
- If prompt is very long / dense code: choose qwen.
- Otherwise: choose doubao.
"""


def fallback_rule(operator: str, prompt_text: str) -> Tuple[str, float, str]:
    tl = prompt_text.lower()
    if any(k in tl for k in _ERROR_KW):
        return "deepseek", 0.2, "fallback:error"
    if operator in ("M1", "M2"):
        return "deepseek", 0.2, f"fallback:{operator}"
    if operator == "E2":
        return "qwen", 0.2, "fallback:E2"
    if operator == "E1":
        return "doubao", 0.2, "fallback:E1"
    return "doubao", 0.2, "fallback:default"


def build_router_prompt(policy_text: str, operator: str, prompt_text: str) -> str:
    backend_cards = (
        "- deepseek: strong debugging and patching; good for fixing failing code and local edits.\n"
        "- qwen: strong restructuring and long-context handling; good for global rewrites.\n"
        "- doubao: low-cost fast exploration; good for generating variants.\n"
    )
    return f"""You are a router that selects ONE backend LLM for the next code-generation call.

Backends:
{backend_cards}

Current routing policy (editable variable):
{policy_text}

Context:
- Operator: {operator}
- User prompt (truncated):
\"\"\"{truncate(prompt_text, 5000)}\"\"\"

Return STRICT JSON only (no markdown, no extra text):
{{
  "backend": "deepseek" | "qwen" | "doubao",
  "confidence": 0.0-1.0,
  "reason": "short reason"
}}
"""


def build_textgrad_prompt(
    policy_text: str,
    batch: List[Dict[str, Any]],
    objective: str,
) -> str:
    """
    critic 生成 textual gradient：输出 new_policy + rationale + changes
    """
    batch_json = json.dumps(batch, ensure_ascii=False, indent=2)
    return f"""You are a TextGrad critic optimizing a routing policy (text variable) for a multi-LLM router.

Objective:
{objective}

Current policy text:
\"\"\"{policy_text}\"\"\"

Recent training batch (JSON):
{batch_json}

Task:
- Analyze failure patterns and which backends work better per operator/context.
- Propose an improved policy text that is SHORT, structured, and deterministic.
- Keep the policy format similar. Only change what is necessary.

Return STRICT JSON only:
{{
  "new_policy": "full updated policy text",
  "rationale": "why this improves objective",
  "changes": ["bullet change 1", "bullet change 2", ...]
}}
"""


# =========================
# TextGrad Router LLM
# =========================
class TextGradRouterLLM:
    """
    - router_llm: 做“路由决策”的 LLM（建议便宜稳定，temperature=0）
    - critic_llm: 做“文本梯度/策略更新”的 LLM（可用更强一点）
    - backends: deepseek/qwen/doubao
    """

    def __init__(
        self,
        router_llm: HttpsApi,
        critic_llm: HttpsApi,
        deepseek_llm: HttpsApi,
        qwen_llm: HttpsApi,
        doubao_llm: HttpsApi,
        policy_text: str = DEFAULT_POLICY,
        tracer: Optional[TGTracer] = None,
        verbose: bool = False,
        # update controls
        freeze: bool = False,
        update_every: int = 25,        # 每收集多少条 outcome 做一次 TextGrad 更新
        batch_size: int = 25,          # 送给 critic 的 batch 大小（<=update_every）
        max_updates: int = 50,         # 训练期间最多更新次数（防止爆预算）
        reward_shift: float = 6000.0,  # 防止“未更新=0”在某些策略里误导，常数平移
    ):
        self.router_llm = router_llm
        self.critic_llm = critic_llm
        self.deepseek_llm = deepseek_llm
        self.qwen_llm = qwen_llm
        self.doubao_llm = doubao_llm

        self.policy_text = policy_text
        self.policy_version = 0
        self.tracer = tracer
        self.verbose = verbose

        self.freeze = freeze
        self.update_every = int(update_every)
        self.batch_size = int(batch_size)
        self.max_updates = int(max_updates)
        self.reward_shift = float(reward_shift)

        self._step = 0
        self._updates = 0

        self._tl = threading.local()
        self._buffer: List[Dict[str, Any]] = []   # outcomes for critic
        self._history: List[Dict[str, Any]] = []  # policy updates history

        self._lock = threading.Lock()

    # ---- threading glue ----
    def _set_last_decision(self, d: TGDecision):
        self._tl.last_decision = d

    def pop_last_decision(self) -> Optional[TGDecision]:
        d = getattr(self._tl, "last_decision", None)
        if hasattr(self._tl, "last_decision"):
            delattr(self._tl, "last_decision")
        return d

    # ---- router forward ----
    def _pick_backend_obj(self, backend: str) -> HttpsApi:
        if backend == "deepseek":
            return self.deepseek_llm
        if backend == "qwen":
            return self.qwen_llm
        return self.doubao_llm

    def route(self, operator: str, prompt_text: str) -> Tuple[str, float, str]:
        rp = build_router_prompt(self.policy_text, operator, prompt_text)
        out = self.router_llm.draw_sample(rp)
        obj = safe_json_loads(out)
        if obj:
            backend = str(obj.get("backend", "")).strip().lower()
            reason = str(obj.get("reason", "")).strip()
            try:
                conf = float(obj.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            if backend in ("deepseek", "qwen", "doubao"):
                return backend, conf, reason[:300]
        return fallback_rule(operator, prompt_text)

    def draw_sample(self, prompt: Any, *args, **kwargs) -> Any:
        self._step += 1
        text = prompt_to_text(prompt)
        operator = detect_operator(text)
        sha = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

        backend, conf, reason = self.route(operator, text)
        decision = TGDecision(
            step=self._step,
            operator=operator,
            prompt_sha256=sha,
            backend=backend,
            confidence=conf,
            reason=reason,
            policy_version=self.policy_version,
        )
        self._set_last_decision(decision)

        if self.verbose:
            print(f"[TextGradRouter] step={self._step} op={operator} -> {backend} conf={conf:.2f}")

        return self._pick_backend_obj(backend).draw_sample(prompt, *args, **kwargs)

    def draw_samples(self, *args, **kwargs):
        n = kwargs.pop("n", None)
        if n is not None:
            n = int(n)
            return [self.draw_sample(*args, **kwargs) for _ in range(n)]
        return self.draw_sample(*args, **kwargs)

    # ---- observe + textgrad update ----
    def observe(self, decision: TGDecision, ok: bool, score: Optional[float], reward_mode: str, prev_best: float):
        """
        在 EoH 评估之后调用，把 outcome 放入 buffer，并可能触发 textgrad 更新。
        """
        if score is None:
            reward = -1.0
        else:
            s = float(score)
            if reward_mode == "score":
                reward = s
            elif reward_mode == "delta":
                reward = s - float(prev_best)
            else:  # improve
                reward = max(float(prev_best), s) - float(prev_best)

        # 平移：避免 reward 全负导致某些策略比较怪（可设为0禁用）
        reward_shifted = float(reward) + self.reward_shift

        outcome = TGOutcome(
            step=decision.step,
            operator=decision.operator,
            backend=decision.backend,
            ok=bool(ok),
            score=None if score is None else float(score),
            reward=reward_shifted,
            policy_version=decision.policy_version,
        )

        if self.tracer is not None:
            try:
                self.tracer.log(decision, outcome)
            except Exception:
                pass

        # 冻结就只记录，不更新
        if self.freeze:
            return

        with self._lock:
            self._buffer.append({
                "step": outcome.step,
                "operator": outcome.operator,
                "backend": outcome.backend,
                "ok": outcome.ok,
                "score": outcome.score,
                "reward": outcome.reward,
                "policy_version": outcome.policy_version,
                "reason": decision.reason[:120],
            })

            if self._updates >= self.max_updates:
                return

            if len(self._buffer) >= self.update_every:
                batch = self._buffer[-self.batch_size:]
                self._do_textgrad_update(batch)

    def _do_textgrad_update(self, batch: List[Dict[str, Any]]):
        objective = (
            "Maximize the EoH evaluation score (higher is better). "
            "The router should pick the backend that yields better scores under each operator/context."
        )
        tg_prompt = build_textgrad_prompt(self.policy_text, batch, objective)
        out = self.critic_llm.draw_sample(tg_prompt)
        obj = safe_json_loads(out)

        if not obj or "new_policy" not in obj:
            # 如果 critic 输出不合格，跳过这次更新
            self._updates += 1
            self._history.append({
                "time": time.time(),
                "ok": False,
                "policy_version_before": self.policy_version,
                "error": "critic_output_not_json",
                "raw": out[:400],
            })
            return

        new_policy = str(obj.get("new_policy", "")).strip()
        rationale = str(obj.get("rationale", "")).strip()
        changes = obj.get("changes", [])

        # 简单安全检查：太短/太长都拒绝
        if len(new_policy) < 80 or len(new_policy) > 4000:
            self._updates += 1
            self._history.append({
                "time": time.time(),
                "ok": False,
                "policy_version_before": self.policy_version,
                "error": "new_policy_length_bad",
                "raw": out[:400],
            })
            return

        old_policy = self.policy_text
        self.policy_text = new_policy
        self.policy_version += 1
        self._updates += 1

        self._history.append({
            "time": time.time(),
            "ok": True,
            "policy_version_after": self.policy_version,
            "rationale": rationale[:500],
            "changes": changes if isinstance(changes, list) else [str(changes)],
            "batch_size": len(batch),
            "old_policy_sha256": hashlib.sha256(old_policy.encode("utf-8")).hexdigest(),
            "new_policy_sha256": hashlib.sha256(new_policy.encode("utf-8")).hexdigest(),
        })

        if self.verbose:
            print(f"[TextGradRouter] policy updated -> v{self.policy_version} (batch={len(batch)})")

    # ---- save/load ----
    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "policy_text": self.policy_text,
            "updates": self._updates,
            "history": self._history,
            "update_every": self.update_every,
            "batch_size": self.batch_size,
            "max_updates": self.max_updates,
            "reward_shift": self.reward_shift,
        }

    def save_state(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_state(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def close(self):
        for b in (self.router_llm, self.critic_llm, self.deepseek_llm, self.qwen_llm, self.doubao_llm):
            if hasattr(b, "close"):
                try:
                    b.close()
                except Exception:
                    pass