from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llm4ad.tools.llm.llm_api_https import HttpsApi


# =========================================================
# Prompt parsing + feature extraction (Signals)
# =========================================================
_OP_RE = re.compile(r"(operator|op)\s*[:=]\s*(E1|E2|M1|M2)\b", flags=re.IGNORECASE)
_OP_ANY_RE = re.compile(r"\b(E1|E2|M1|M2)\b")

_ERROR_KW = [
    "traceback", "exception", "syntaxerror", "typeerror", "nameerror",
    "valueerror", "indexerror", "keyerror", "assertionerror", "indentationerror",
]

_FIX_WORDS = ["fix", "bug", "repair", "patch", "debug", "crash", "error", "exception", "modify", "edit", "refine", "correct"]
_EXPLORE_WORDS = ["design", "propose", "invent", "rewrite", "new heuristic", "explore", "diversify", "variant", "generate"]


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


def compute_signals(prompt_text: str) -> Dict[str, Any]:
    tl = prompt_text.lower()
    op = detect_operator(prompt_text)

    n_chars = len(prompt_text)
    n_lines = len(prompt_text.splitlines())

    # crude code density markers
    markers = 0
    for pat in ["def ", "class ", "```", "import ", "return ", " for ", " while ", "np.", "numpy", "torch", "sklearn"]:
        markers += tl.count(pat)
    code_density = min(1.0, markers / 25.0)

    has_error = any(k in tl for k in _ERROR_KW)
    fix_score = sum(1 for w in _FIX_WORDS if w in tl)
    explore_score = sum(1 for w in _EXPLORE_WORDS if w in tl)

    return {
        "operator": op,
        "n_chars": n_chars,
        "n_lines": n_lines,
        "code_density": round(code_density, 3),
        "has_error": bool(has_error),
        "fix_score": int(fix_score),
        "explore_score": int(explore_score),
        "markers": int(markers),
    }


# =========================================================
# Model profiles (capability cards)
# =========================================================
@dataclass
class ModelProfile:
    name: str
    strengths: List[str]
    weaknesses: List[str]
    cost_hint: str  # "low/medium/high"
    reliability_hint: str  # "low/medium/high"

    def to_text(self) -> str:
        s = ", ".join(self.strengths)
        w = ", ".join(self.weaknesses)
        return f"- {self.name}: strengths=[{s}]; weaknesses=[{w}]; cost={self.cost_hint}; reliability={self.reliability_hint}"


def build_profiles(profile_mode: str) -> Tuple[List[ModelProfile], Dict[str, str]]:
    """
    profile_mode:
      - "named": output backend names: deepseek/qwen/doubao
      - "anonymous": output A/B/C to reduce brand bias; map labels->backend
    """
    if profile_mode == "anonymous":
        # Fixed mapping to keep reproducible
        label_to_backend = {"A": "deepseek", "B": "qwen", "C": "doubao"}
        profiles = [
            ModelProfile("A", ["debugging", "patching", "local edits"], ["less exploratory"], "medium", "high"),
            ModelProfile("B", ["global restructuring", "long-context"], ["sometimes less strict"], "medium", "medium"),
            ModelProfile("C", ["fast exploration", "low cost", "generate variants"], ["can be unstable"], "low", "medium"),
        ]
        return profiles, label_to_backend

    # named
    profiles = [
        ModelProfile("deepseek", ["debugging", "patching", "local edits"], ["less exploratory"], "medium", "high"),
        ModelProfile("qwen", ["global restructuring", "long-context"], ["may over-edit"], "medium", "medium"),
        ModelProfile("doubao", ["fast exploration", "low cost", "generate variants"], ["can be unstable"], "low", "medium"),
    ]
    return profiles, {}


# =========================================================
# Prior stats (operator x backend) used as frozen prior in test
# =========================================================
@dataclass
class Bucket:
    count: int = 0
    sum_score: float = 0.0
    fail_count: int = 0

    def mean_score(self) -> float:
        return self.sum_score / self.count if self.count > 0 else 0.0


class PriorStats:
    OPS = ["E1", "E2", "M1", "M2", "UNKNOWN"]
    BACKENDS = ["deepseek", "qwen", "doubao"]

    def __init__(self):
        self.stats: Dict[str, Dict[str, Bucket]] = {
            op: {b: Bucket() for b in self.BACKENDS} for op in self.OPS
        }

    def update(self, operator: str, backend: str, ok: bool, score: Optional[float]):
        op = operator if operator in self.stats else "UNKNOWN"
        b = backend if backend in self.stats[op] else "deepseek"
        bk = self.stats[op][b]
        bk.count += 1
        if score is not None:
            bk.sum_score += float(score)
        if not ok:
            bk.fail_count += 1

    def summary_text(self, operator: str) -> str:
        op = operator if operator in self.stats else "UNKNOWN"
        parts = []
        for b in self.BACKENDS:
            bk = self.stats[op][b]
            parts.append(f"{b}: n={bk.count}, meanS={bk.mean_score():.2f}, fail={bk.fail_count}")
        return f"{op}: " + " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for op, mm in self.stats.items():
            out[op] = {}
            for b, bk in mm.items():
                out[op][b] = dataclasses.asdict(bk)
        return out

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PriorStats":
        ps = PriorStats()
        for op, mm in d.items():
            if op not in ps.stats:
                continue
            for b, bd in mm.items():
                if b not in ps.stats[op]:
                    continue
                ps.stats[op][b] = Bucket(
                    count=int(bd.get("count", 0)),
                    sum_score=float(bd.get("sum_score", 0.0)),
                    fail_count=int(bd.get("fail_count", 0)),
                )
        return ps


# =========================================================
# Trace
# =========================================================
@dataclass
class TGDecision:
    step: int
    operator: str
    prompt_sha256: str
    backend: str
    confidence: float
    reason: str
    policy_version: int
    signals: Dict[str, Any]
    backend_label: str  # for anonymous mode, "A/B/C", otherwise same as backend


@dataclass
class TGOutcome:
    step: int
    ok: bool
    score: Optional[float]
    reward: float
    policy_version: int


class TGTracer:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step", "operator", "backend", "backend_label",
                        "confidence", "reason", "policy_version",
                        "ok", "score", "reward",
                        "signals_json", "prompt_sha256", "time"
                    ],
                )
                w.writeheader()

    def log(self, decision: TGDecision, outcome: TGOutcome):
        row = {
            "step": decision.step,
            "operator": decision.operator,
            "backend": decision.backend,
            "backend_label": decision.backend_label,
            "confidence": f"{decision.confidence:.3f}",
            "reason": decision.reason[:200].replace("\n", " "),
            "policy_version": decision.policy_version,
            "ok": str(bool(outcome.ok)),
            "score": "" if outcome.score is None else f"{float(outcome.score):.6f}",
            "reward": f"{float(outcome.reward):.6f}",
            "signals_json": json.dumps(decision.signals, ensure_ascii=False),
            "prompt_sha256": decision.prompt_sha256,
            "time": f"{time.time():.3f}",
        }
        with self._lock:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step", "operator", "backend", "backend_label",
                        "confidence", "reason", "policy_version",
                        "ok", "score", "reward",
                        "signals_json", "prompt_sha256", "time"
                    ],
                )
                w.writerow(row)


# =========================================================
# Policy text + TextGrad prompts
# =========================================================
DEFAULT_POLICY = """[Routing Policy v0: Multi-signal Scorecard]
Goal: choose the backend that maximizes expected OBP score (higher is better), considering correctness and cost.

Signals (from input):
- operator ∈ {E1,E2,M1,M2,UNKNOWN}
- has_error (bool), n_chars, n_lines, code_density
- fix_score, explore_score
- prior_stats: per-operator mean score and fail counts for each backend (frozen in test)

Decision procedure:
1) Hard constraints:
   - if has_error=true -> strongly prefer debugging-focused backend.
2) Score each backend using a weighted scorecard:
   Score = w_op + w_error + w_len + w_density + w_fix + w_explore + w_prior + w_cost
   (weights described below).
3) Choose the backend with the highest Score.
4) If top-2 scores are close, choose the cheaper/more reliable backend and set confidence low.

Initial weights / preferences (editable):
- debugging/fix signals favor deepseek
- long-context/global restructure favors qwen
- exploration favors doubao
- incorporate prior_stats: prefer higher meanS and lower fail
- keep cost low when confidence is low (prefer doubao)

Output JSON with backend + confidence + reason citing 2-3 signals.
"""


def build_router_prompt(
    profiles: List[ModelProfile],
    profile_mode: str,
    policy_text: str,
    signals: Dict[str, Any],
    prior_summary: str,
    prompt_text: str,
) -> str:
    profiles_text = "\n".join([p.to_text() for p in profiles])

    # output space differs for anonymous mode
    if profile_mode == "anonymous":
        backend_space = '"A" | "B" | "C"'
    else:
        backend_space = '"deepseek" | "qwen" | "doubao"'

    return f"""You are a routing controller. Choose ONE backend LLM for the next code-generation call.

Backends (capability cards):
{profiles_text}

Current routing policy (text variable optimized by TextGrad):
\"\"\"{policy_text}\"\"\"

Signals (JSON):
{json.dumps(signals, ensure_ascii=False)}

Prior stats summary (for current operator; higher meanS is better, lower fail is better):
{prior_summary}

User prompt excerpt:
\"\"\"{truncate(prompt_text, 3500)}\"\"\"

Return STRICT JSON only (no markdown, no extra text):
{{
  "backend": {backend_space},
  "confidence": 0.0-1.0,
  "reason": "cite the key signals and profile traits"
}}
"""


def build_textgrad_prompt(
    profiles: List[ModelProfile],
    policy_text: str,
    batch: List[Dict[str, Any]],
) -> str:
    profiles_text = "\n".join([p.to_text() for p in profiles])
    batch_json = json.dumps(batch, ensure_ascii=False, indent=2)
    return f"""You are a TextGrad critic optimizing a routing policy (text variable) for a multi-LLM router.

Objective:
- Maximize expected OBP score (higher is better).
- Reduce catastrophic failures (very low scores / frequent failures).
- Keep policy interpretable: multi-signal scorecard + clear tie-breaking.

Backends (capability cards):
{profiles_text}

Current policy:
\"\"\"{policy_text}\"\"\"

Recent batch (JSON). Each item includes signals, chosen backend, score, ok:
{batch_json}

Task:
1) Diagnose which signals correlate with better backend choices.
2) Propose a revised policy that uses MORE THAN operator only (must reference at least 3 other signals).
3) Keep it short and structured; keep the "Decision procedure" format.
4) Add/adjust weights, thresholds, and tie-break rules to improve robustness.

Return STRICT JSON only:
{{
  "new_policy": "full updated policy text",
  "rationale": "why this improves objective",
  "changes": ["change 1", "change 2", ...]
}}
"""


def policy_guardrail_ok(new_policy: str, profile_mode: str) -> bool:
    if not (120 <= len(new_policy) <= 5000):
        return False
    must = ["Decision procedure", "Signals", "Output JSON"]
    for m in must:
        if m not in new_policy:
            return False
    if profile_mode == "anonymous":
        # ensure it mentions A/B/C at least once
        if not any(x in new_policy for x in ["A", "B", "C"]):
            return False
    else:
        if not any(x in new_policy for x in ["deepseek", "qwen", "doubao"]):
            return False
    return True


# =========================================================
# TextGrad Router
# =========================================================
class TextGradRouterLLM:
    """
    - router_llm: per-call routing decision model (temperature=0 recommended)
    - critic_llm: generates text gradients to update policy
    - prior_stats: updated in train, frozen in test (used as prior_summary in prompt)
    """
    def __init__(
        self,
        router_llm: HttpsApi,
        critic_llm: HttpsApi,
        deepseek_llm: HttpsApi,
        qwen_llm: HttpsApi,
        doubao_llm: HttpsApi,
        profile_mode: str = "named",
        policy_text: str = DEFAULT_POLICY,
        tracer: Optional[TGTracer] = None,
        verbose: bool = False,
        freeze: bool = False,
        reward_mode: str = "score",
        update_every: int = 25,
        batch_size: int = 25,
        max_updates: int = 30,
        reward_shift: float = 6000.0,
        verify_update: bool = False,  # optional extra LLM check
    ):
        self.router_llm = router_llm
        self.critic_llm = critic_llm
        self.deepseek_llm = deepseek_llm
        self.qwen_llm = qwen_llm
        self.doubao_llm = doubao_llm

        self.profile_mode = profile_mode
        self.profiles, self.label_to_backend = build_profiles(profile_mode)
        self.policy_text = policy_text
        self.policy_version = 0

        self.tracer = tracer
        self.verbose = verbose
        self.freeze = freeze
        self.reward_mode = reward_mode

        self.update_every = int(update_every)
        self.batch_size = int(batch_size)
        self.max_updates = int(max_updates)
        self.reward_shift = float(reward_shift)
        self.verify_update = bool(verify_update)

        self._step = 0
        self._updates = 0
        self._history: List[Dict[str, Any]] = []
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        self.prior_stats = PriorStats()

        self._tl = threading.local()  # for pop_last_decision

    def _set_last_decision(self, d: TGDecision):
        self._tl.last_decision = d

    def pop_last_decision(self) -> Optional[TGDecision]:
        d = getattr(self._tl, "last_decision", None)
        if hasattr(self._tl, "last_decision"):
            delattr(self._tl, "last_decision")
        return d

    def _pick_backend_obj(self, backend: str) -> HttpsApi:
        if backend == "deepseek":
            return self.deepseek_llm
        if backend == "qwen":
            return self.qwen_llm
        return self.doubao_llm

    def _fallback(self, signals: Dict[str, Any], prompt_text: str) -> Tuple[str, float, str, str]:
        op = str(signals.get("operator", "UNKNOWN"))
        tl = prompt_text.lower()
        if any(k in tl for k in _ERROR_KW) or bool(signals.get("has_error", False)):
            return "deepseek", 0.2, "fallback:error", ("A" if self.profile_mode == "anonymous" else "deepseek")
        if op in ("M1", "M2"):
            return "deepseek", 0.2, f"fallback:{op}", ("A" if self.profile_mode == "anonymous" else "deepseek")
        if op == "E2":
            return "qwen", 0.2, "fallback:E2", ("B" if self.profile_mode == "anonymous" else "qwen")
        if op == "E1":
            return "doubao", 0.2, "fallback:E1", ("C" if self.profile_mode == "anonymous" else "doubao")
        return "doubao", 0.2, "fallback:default", ("C" if self.profile_mode == "anonymous" else "doubao")

    def route(self, signals: Dict[str, Any], prompt_text: str) -> Tuple[str, float, str, str]:
        op = str(signals.get("operator", "UNKNOWN"))
        prior_summary = self.prior_stats.summary_text(op)

        rp = build_router_prompt(
            profiles=self.profiles,
            profile_mode=self.profile_mode,
            policy_text=self.policy_text,
            signals=signals,
            prior_summary=prior_summary,
            prompt_text=prompt_text,
        )
        out = self.router_llm.draw_sample(rp)
        obj = safe_json_loads(out)
        if not obj:
            return self._fallback(signals, prompt_text)

        backend_raw = str(obj.get("backend", "")).strip()
        reason = str(obj.get("reason", "")).strip()
        try:
            conf = float(obj.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        if self.profile_mode == "anonymous":
            # expect A/B/C
            if backend_raw in self.label_to_backend:
                backend = self.label_to_backend[backend_raw]
                return backend, conf, reason[:300], backend_raw
            return self._fallback(signals, prompt_text)

        # named
        backend = backend_raw.lower()
        if backend in ("deepseek", "qwen", "doubao"):
            return backend, conf, reason[:300], backend
        return self._fallback(signals, prompt_text)

    def draw_sample(self, prompt: Any, *args, **kwargs) -> Any:
        self._step += 1
        prompt_text = prompt_to_text(prompt)
        signals = compute_signals(prompt_text)
        op = str(signals["operator"])

        sha = hashlib.sha256(prompt_text.encode("utf-8", errors="ignore")).hexdigest()
        backend, conf, reason, backend_label = self.route(signals, prompt_text)

        decision = TGDecision(
            step=self._step,
            operator=op,
            prompt_sha256=sha,
            backend=backend,
            confidence=conf,
            reason=reason,
            policy_version=self.policy_version,
            signals=signals,
            backend_label=backend_label,
        )
        self._set_last_decision(decision)

        if self.verbose:
            print(f"[TextGradRouter] step={self._step} op={op} -> {backend} conf={conf:.2f}")

        return self._pick_backend_obj(backend).draw_sample(prompt, *args, **kwargs)

    def draw_samples(self, *args, **kwargs):
        n = kwargs.pop("n", None)
        if n is not None:
            n = int(n)
            return [self.draw_sample(*args, **kwargs) for _ in range(n)]
        return self.draw_sample(*args, **kwargs)

    def _compute_reward(self, prev_best: float, score: Optional[float]) -> float:
        if score is None:
            return -1.0
        s = float(score)
        if self.reward_mode == "score":
            r = s
        elif self.reward_mode == "delta":
            r = s - float(prev_best)
        else:  # improve
            r = max(float(prev_best), s) - float(prev_best)
        return float(r) + self.reward_shift

    def observe(self, decision: TGDecision, ok: bool, score: Optional[float], prev_best: float):
        """
        Called by EoH after evaluation. In test, freeze=True so it logs but does not update policy/stats.
        """
        reward = self._compute_reward(prev_best=prev_best, score=score)

        # update prior stats only in training
        if not self.freeze:
            self.prior_stats.update(decision.operator, decision.backend, ok=ok, score=score)

        outcome = TGOutcome(
            step=decision.step,
            ok=bool(ok),
            score=None if score is None else float(score),
            reward=float(reward),
            policy_version=decision.policy_version,
        )

        if self.tracer is not None:
            try:
                self.tracer.log(decision, outcome)
            except Exception:
                pass

        if self.freeze:
            return

        with self._lock:
            self._buffer.append({
                "step": decision.step,
                "operator": decision.operator,
                "backend": decision.backend_label if self.profile_mode == "anonymous" else decision.backend,
                "ok": bool(ok),
                "score": None if score is None else float(score),
                "reward": float(reward),
                "signals": decision.signals,
                "reason": decision.reason[:160],
                "policy_version": decision.policy_version,
            })

            if self._updates >= self.max_updates:
                return
            if len(self._buffer) >= self.update_every:
                batch = self._buffer[-self.batch_size:]
                self._do_textgrad_update(batch)

    def _verify_policy(self, old_policy: str, new_policy: str, batch: List[Dict[str, Any]]) -> bool:
        """
        Optional safety check: ask router_llm to choose which policy is better on the batch.
        Keeps cost low by using one call occasionally.
        """
        try:
            payload = {
                "old_policy": old_policy,
                "new_policy": new_policy,
                "batch": batch[: min(10, len(batch))],
                "question": "Which policy is more likely to maximize score and reduce failures? Reply JSON {\"better\":\"old\"|\"new\",\"reason\":\"...\"}."
            }
            prompt = "Evaluate routing policies. Reply STRICT JSON only.\n" + json.dumps(payload, ensure_ascii=False)
            out = self.router_llm.draw_sample(prompt)
            obj = safe_json_loads(out)
            if obj and obj.get("better") in ("old", "new"):
                return obj["better"] == "new"
        except Exception:
            pass
        return True  # fail-open

    def _do_textgrad_update(self, batch: List[Dict[str, Any]]):
        tg_prompt = build_textgrad_prompt(self.profiles, self.policy_text, batch)
        out = self.critic_llm.draw_sample(tg_prompt)
        obj = safe_json_loads(out)

        self._updates += 1

        if not obj or "new_policy" not in obj:
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

        if not policy_guardrail_ok(new_policy, self.profile_mode):
            self._history.append({
                "time": time.time(),
                "ok": False,
                "policy_version_before": self.policy_version,
                "error": "policy_guardrail_failed",
                "raw": out[:400],
            })
            return

        old_policy = self.policy_text

        if self.verify_update and not self._verify_policy(old_policy, new_policy, batch):
            self._history.append({
                "time": time.time(),
                "ok": False,
                "policy_version_before": self.policy_version,
                "error": "verify_rejected",
                "rationale": rationale[:300],
            })
            return

        self.policy_text = new_policy
        self.policy_version += 1
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
            print(f"[TextGradRouter] policy updated -> v{self.policy_version}")

    # ----- save/load -----
    def state_dict(self) -> Dict[str, Any]:
        return {
            "profile_mode": self.profile_mode,
            "policy_version": self.policy_version,
            "policy_text": self.policy_text,
            "updates": self._updates,
            "history": self._history,
            "update_every": self.update_every,
            "batch_size": self.batch_size,
            "max_updates": self.max_updates,
            "reward_mode": self.reward_mode,
            "reward_shift": self.reward_shift,
            "prior_stats": self.prior_stats.to_dict(),
            "verify_update": self.verify_update,
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