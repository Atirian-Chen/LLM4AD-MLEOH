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
import random
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


def _strip_code_fence(s: str) -> str:
    # remove ```json ... ``` or ``` ... ```
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def extract_json_obj(s: str) -> Optional[str]:
    if not s:
        return None
    s = _strip_code_fence(s)
    a = s.find("{")
    b = s.rfind("}")
    if a == -1 or b == -1 or b <= a:
        return None
    return s[a:b + 1]


def safe_json_loads(s: str) -> Optional[Dict[str, Any]]:
    js = extract_json_obj(s)
    if js is None:
        return None
    # try strict
    try:
        return json.loads(js)
    except Exception:
        pass
    # try to fix common issues: trailing commas
    try:
        js2 = re.sub(r",\s*}", "}", js)
        js2 = re.sub(r",\s*]", "]", js2)
        return json.loads(js2)
    except Exception:
        return None


def compute_signals(prompt_text: str) -> Dict[str, Any]:
    tl = prompt_text.lower()
    op = detect_operator(prompt_text)

    n_chars = len(prompt_text)
    n_lines = len(prompt_text.splitlines())

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
    cost_hint: str
    reliability_hint: str

    def to_text(self) -> str:
        s = ", ".join(self.strengths)
        w = ", ".join(self.weaknesses)
        return f"- {self.name}: strengths=[{s}]; weaknesses=[{w}]; cost={self.cost_hint}; reliability={self.reliability_hint}"


def build_profiles(profile_mode: str) -> Tuple[List[ModelProfile], Dict[str, str]]:
    if profile_mode == "anonymous":
        label_to_backend = {"A": "deepseek", "B": "qwen", "C": "doubao"}
        profiles = [
            ModelProfile("A", ["debugging", "patching", "local edits"], ["less exploratory"], "medium", "high"),
            ModelProfile("B", ["global restructuring", "long-context"], ["sometimes less strict"], "medium", "medium"),
            ModelProfile("C", ["fast exploration", "low cost", "generate variants"], ["can be unstable"], "low", "medium"),
        ]
        return profiles, label_to_backend

    profiles = [
        ModelProfile("deepseek", ["debugging", "patching", "local edits"], ["less exploratory"], "medium", "high"),
        ModelProfile("qwen", ["global restructuring", "long-context"], ["may over-edit"], "medium", "medium"),
        ModelProfile("doubao", ["fast exploration", "low cost", "generate variants"], ["can be unstable"], "low", "medium"),
    ]
    return profiles, {}


# =========================================================
# Running stats for score normalization (per-operator)
# =========================================================
@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0  # sum of squares of differences

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def var(self) -> float:
        if self.n < 2:
            return 1.0
        return self.m2 / (self.n - 1)

    def std(self) -> float:
        return math.sqrt(max(1e-9, self.var()))

    def z(self, x: float) -> float:
        return (x - self.mean) / self.std()


# =========================================================
# Prior stats (operator x backend)
# =========================================================
@dataclass
class Bucket:
    count: int = 0
    sum_score: float = 0.0
    fail_count: int = 0

    def mean_score(self) -> float:
        return self.sum_score / self.count if self.count > 0 else 0.0

    def fail_rate(self) -> float:
        return self.fail_count / self.count if self.count > 0 else 0.0


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

    def bucket(self, operator: str, backend: str) -> Bucket:
        op = operator if operator in self.stats else "UNKNOWN"
        b = backend if backend in self.stats[op] else "deepseek"
        return self.stats[op][b]

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
    backend_label: str


@dataclass
class TGOutcome:
    step: int
    ok: bool
    score: Optional[float]
    reward: float
    policy_version: int
    improve: float
    prev_best: float


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
                        "improve", "prev_best",
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
            "improve": f"{float(outcome.improve):.6f}",
            "prev_best": f"{float(outcome.prev_best):.6f}",
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
                        "improve", "prev_best",
                        "signals_json", "prompt_sha256", "time"
                    ],
                )
                w.writerow(row)


# =========================================================
# Policy text + TextGrad prompts (v3: stricter + shorter)
# =========================================================
DEFAULT_POLICY_V3 = """[Routing Policy v0: Hard constraints + Scorecard + Tie-break]
Goal:
- Maximize the FINAL best OBP score found within budget (best-of-run).
- Avoid catastrophic failures (very low scores / frequent failures).
- Keep routing interpretable.

Signals (from input):
- operator ∈ {E1,E2,M1,M2,UNKNOWN}
- has_error (bool), n_chars, n_lines, code_density, markers
- fix_score, explore_score
- prior_stats (per-operator mean score and fail count per backend; higher mean is better, lower fail is better)

Decision procedure:
1) Hard constraints:
   - If has_error=true -> prefer debugging-strong backend.
   - If very long or very code-dense -> prefer long-context backend.
2) Scorecard:
   For each backend compute:
   Score = Wop + Wlen + Wdensity + Wfix + Wexplore + Wprior + Wrisk + Wcost
   - Wprior: prefer higher mean_score and lower fail_count in prior_stats for this operator.
   - Wrisk: strongly penalize backends with high fail_count or catastrophic history for this operator.
3) Tie-break (when top-2 scores close):
   - Prefer lower fail_count, then higher mean_score, then cheaper backend if confidence is low.
4) Output JSON: {backend, confidence, reason}.

Output JSON (STRICT):
{"backend": "deepseek"|"qwen"|"doubao", "confidence": 0.0-1.0, "reason": "cite 2-3 signals and a profile trait"}.
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
    if profile_mode == "anonymous":
        backend_space = '"A" | "B" | "C"'
    else:
        backend_space = '"deepseek" | "qwen" | "doubao"'

    return f"""You are a routing controller. Choose ONE backend LLM for the next code-generation call.

Backends (capability cards):
{profiles_text}

Current routing policy (optimized text):
\"\"\"{policy_text}\"\"\"

Signals (JSON):
{json.dumps(signals, ensure_ascii=False)}

Prior stats summary (for current operator; higher meanS better; lower fail better):
{prior_summary}

User prompt excerpt:
\"\"\"{truncate(prompt_text, 3200)}\"\"\"

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
    max_policy_chars: int,
) -> str:
    profiles_text = "\n".join([p.to_text() for p in profiles])
    batch_json = json.dumps(batch, ensure_ascii=False, indent=2)

    return f"""You are a TextGrad critic optimizing a routing policy text for a multi-LLM router.

Objective (most important first):
1) Maximize FINAL best-of-run OBP score under a fixed budget (best_score matters).
2) Control catastrophic failures, BUT do not eliminate a backend solely due to moderate failure rate
   if it has higher upside (higher best_score/improvement events). It's acceptable to spend a small
   exploration budget on a risky but high-upside backend to improve final best-of-run.
3) Keep policy interpretable: Hard constraints + Scorecard + Tie-break + Output JSON.

Backends (capability cards):
{profiles_text}

Current policy:
\"\"\"{policy_text}\"\"\"

Recent batch (JSON). Each item includes signals, chosen backend, ok, score, reward, improve, prev_best:
{batch_json}

Task:
- Diagnose which signals correlate with improvement events (improve>0) and high scores.
- Add explicit rules to prevent catastrophic / high-fail backends.
- Use MORE THAN operator only (must reference at least 3 other signals).
- Keep policy concise (<= {max_policy_chars} chars).

Return STRICT JSON only:
{{
  "new_policy": "full updated policy text",
  "rationale": "why this improves best-of-run and reduces failures",
  "changes": ["change 1", "change 2", ...]
}}
"""


def policy_guardrail_ok(new_policy: str, profile_mode: str, min_chars: int, max_chars: int) -> bool:
    if not (min_chars <= len(new_policy) <= max_chars):
        return False
    must = ["Decision procedure", "Signals", "Output JSON"]
    for m in must:
        if m not in new_policy:
            return False
    if profile_mode == "anonymous":
        if not any(x in new_policy for x in ["A", "B", "C"]):
            return False
    else:
        if not any(x in new_policy for x in ["deepseek", "qwen", "doubao"]):
            return False
    return True


# =========================================================
# TextGrad Router v3
# =========================================================
class TextGradRouterLLM:
    """
    v3 improvements:
    - mixed reward: z(score) + normalized_improve + exploration bonus + failure/catastrophic penalty
    - epsilon/UCB exploration (reproducible, not relying on router temperature)
    - balanced batch sampling
    - stricter critic output constraints to reduce guardrail failures
    """

    def __init__(
        self,
        router_llm: HttpsApi,
        critic_llm: HttpsApi,
        deepseek_llm: HttpsApi,
        qwen_llm: HttpsApi,
        doubao_llm: HttpsApi,
        profile_mode: str = "named",
        policy_text: str = DEFAULT_POLICY_V3,
        tracer: Optional[TGTracer] = None,
        verbose: bool = False,
        freeze: bool = False,

        # reward / objective shaping
        reward_alpha: float = 1.0,          # weight for z-score component
        reward_beta: float = 2.0,           # weight for normalized improvement
        reward_gamma: float = 0.5,          # exploration bonus weight
        fail_reward: float = -3.0,          # base reward if score is None / fail
        catastrophic_threshold: float = -4800.0,
        catastrophic_penalty: float = 3.0,
        reward_shift: float = 0.0,          # keep 0 by default in v3

        # TextGrad update cadence
        update_every: int = 25,
        batch_size: int = 25,
        max_updates: int = 30,
        verify_update: bool = False,

        # exploration control (train only)
        explore_epsilon: float = 0.10,
        explore_ucb_c: float = 1.5,
        explore_fail_weight: float = 1200.0,

        # policy guardrail
        min_policy_chars: int = 200,
        max_policy_chars: int = 5000,

        # autosave (optional)
        auto_save_path: Optional[str] = None,
        auto_save_every_updates: int = 1,
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

        # reward shaping
        self.reward_alpha = float(reward_alpha)
        self.reward_beta = float(reward_beta)
        self.reward_gamma = float(reward_gamma)
        self.fail_reward = float(fail_reward)
        self.catastrophic_threshold = float(catastrophic_threshold)
        self.catastrophic_penalty = float(catastrophic_penalty)
        self.reward_shift = float(reward_shift)

        # update
        self.update_every = int(update_every)
        self.batch_size = int(batch_size)
        self.max_updates = int(max_updates)
        self.verify_update = bool(verify_update)

        # exploration
        self.explore_epsilon = float(explore_epsilon)
        self.explore_ucb_c = float(explore_ucb_c)
        self.explore_fail_weight = float(explore_fail_weight)

        # guardrail
        self.min_policy_chars = int(min_policy_chars)
        self.max_policy_chars = int(max_policy_chars)

        # autosave
        self.auto_save_path = auto_save_path
        self.auto_save_every_updates = int(auto_save_every_updates)

        self._step = 0
        self._updates = 0
        self._history: List[Dict[str, Any]] = []
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        self.prior_stats = PriorStats()
        self.score_stats: Dict[str, RunningStats] = {op: RunningStats() for op in PriorStats.OPS}

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

    # ---- minimal hard constraints for safety fallback / exploration ----
    def _hard_constraints_backend(self, signals: Dict[str, Any], prompt_text: str) -> Optional[str]:
        tl = prompt_text.lower()
        op = str(signals.get("operator", "UNKNOWN"))
        n_chars = int(signals.get("n_chars", 0))
        code_density = float(signals.get("code_density", 0.0))

        if any(k in tl for k in _ERROR_KW) or bool(signals.get("has_error", False)):
            return "deepseek"
        if n_chars > 4500:
            return "qwen"
        if n_chars > 2200 and code_density > 0.8:
            return "qwen"
        if op == "E2" and n_chars > 2000:
            return "qwen"
        return None

    def _fallback(self, signals: Dict[str, Any], prompt_text: str) -> Tuple[str, float, str, str]:
        forced = self._hard_constraints_backend(signals, prompt_text)
        if forced:
            if self.profile_mode == "anonymous":
                label = {"deepseek": "A", "qwen": "B", "doubao": "C"}[forced]
                return forced, 0.25, "hard_constraint", label
            return forced, 0.25, "hard_constraint", forced

        op = str(signals.get("operator", "UNKNOWN"))
        if op in ("M1", "M2"):
            b = "deepseek"
        elif op == "E2":
            b = "qwen"
        else:
            b = "doubao"

        if self.profile_mode == "anonymous":
            label = {"deepseek": "A", "qwen": "B", "doubao": "C"}[b]
            return b, 0.2, "fallback", label
        return b, 0.2, "fallback", b

    def _ucb_pick(self, operator: str) -> str:
        # choose by (mean score) + c * sqrt(log(total)/ (n+1)) - fail_weight*fail_rate
        total = 0
        for b in PriorStats.BACKENDS:
            total += self.prior_stats.bucket(operator, b).count
        total = max(1, total)

        best_b = "doubao"
        best_v = -1e18
        for b in PriorStats.BACKENDS:
            bk = self.prior_stats.bucket(operator, b)
            mean_s = bk.mean_score()
            n = bk.count
            ucb = self.explore_ucb_c * math.sqrt(math.log(total + 1.0) / (n + 1.0))
            fail_pen = self.explore_fail_weight * bk.fail_rate()
            v = mean_s + ucb - fail_pen
            if v > best_v:
                best_v = v
                best_b = b
        return best_b

    def route(self, signals: Dict[str, Any], prompt_text: str) -> Tuple[str, float, str, str]:
        op = str(signals.get("operator", "UNKNOWN"))
        prior_summary = self.prior_stats.summary_text(op)

        # train-time controlled exploration
        if (not self.freeze) and (self.explore_epsilon > 0.0) and (random.random() < self.explore_epsilon):
            forced = self._hard_constraints_backend(signals, prompt_text)
            backend = forced if forced else self._ucb_pick(op)
            if self.profile_mode == "anonymous":
                label = {"deepseek": "A", "qwen": "B", "doubao": "C"}[backend]
                return backend, 0.15, "epsilon_ucb_explore", label
            return backend, 0.15, "epsilon_ucb_explore", backend

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
            if backend_raw in self.label_to_backend:
                backend = self.label_to_backend[backend_raw]
                return backend, conf, reason[:300], backend_raw
            return self._fallback(signals, prompt_text)

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
            print(f"[TextGradRouterV3] step={self._step} op={op} -> {backend} conf={conf:.2f}")

        return self._pick_backend_obj(backend).draw_sample(prompt, *args, **kwargs)

    def draw_samples(self, *args, **kwargs):
        n = kwargs.pop("n", None)
        if n is not None:
            n = int(n)
            return [self.draw_sample(*args, **kwargs) for _ in range(n)]
        return self.draw_sample(*args, **kwargs)

    # ---- reward aligned with "final best" ----
    def _compute_reward(self, operator: str, backend: str, prev_best: float, score: Optional[float], ok: bool) -> Tuple[float, float]:
        """
        returns (reward, improve)
        improve = max(0, score - prev_best)
        reward combines:
          - z-score(score) per-operator
          - normalized improve (align best-of-run)
          - exploration bonus 1/sqrt(n(op,backend)+1)
          - failure/catastrophic penalty
        """
        if (score is None) or (not ok):
            return self.fail_reward + self.reward_shift, 0.0

        s = float(score)
        improve = max(0.0, s - float(prev_best))

        # z-score uses stats BEFORE update (to avoid leakage)
        rs = self.score_stats.get(operator, self.score_stats["UNKNOWN"])
        z = rs.z(s) if rs.n >= 5 else 0.0
        z = float(max(-5.0, min(5.0, z)))

        # normalized improve (scale-insensitive)
        scale = max(1000.0, abs(float(prev_best)))
        norm_improve = float(improve) / float(scale)

        # exploration bonus based on counts
        n = self.prior_stats.bucket(operator, backend).count
        explore_bonus = 1.0 / math.sqrt(n + 1.0)

        reward = (
            self.reward_alpha * z
            + self.reward_beta * norm_improve
            + self.reward_gamma * explore_bonus
        )

        # catastrophic penalty
        if s <= self.catastrophic_threshold:
            reward -= self.catastrophic_penalty

        reward += self.reward_shift
        return float(reward), float(improve)

    def observe(self, decision: TGDecision, ok: bool, score: Optional[float], prev_best: float):
        """
        Called by EoH after evaluation.
        In test, freeze=True so it logs but does not update policy/stats.
        """
        operator = decision.operator
        backend = decision.backend

        reward, improve = self._compute_reward(operator, backend, prev_best=prev_best, score=score, ok=ok)

        # update priors & score stats only in training
        if not self.freeze:
            self.prior_stats.update(operator, backend, ok=ok, score=score)
            if ok and score is not None:
                self.score_stats.get(operator, self.score_stats["UNKNOWN"]).update(float(score))

        outcome = TGOutcome(
            step=decision.step,
            ok=bool(ok),
            score=None if score is None else float(score),
            reward=float(reward),
            policy_version=decision.policy_version,
            improve=float(improve),
            prev_best=float(prev_best),
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
                "improve": float(improve),
                "prev_best": float(prev_best),
                "signals": decision.signals,
                "reason": decision.reason[:160],
                "policy_version": decision.policy_version,
            })

            if self._updates >= self.max_updates:
                return
            if len(self._buffer) >= self.update_every:
                batch = self._build_balanced_batch(self._buffer, self.batch_size)
                self._do_textgrad_update(batch)

    def _build_balanced_batch(self, buf: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        """
        Balanced sampling across operators (and as much as possible across backends).
        Uses newest-first per group.
        """
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for x in reversed(buf[-2000:]):  # cap memory
            op = str(x.get("operator", "UNKNOWN"))
            be = str(x.get("backend", ""))
            groups.setdefault((op, be), []).append(x)

        # round-robin pick
        keys = list(groups.keys())
        random.shuffle(keys)
        out: List[Dict[str, Any]] = []
        ptr = 0
        while len(out) < batch_size and keys:
            k = keys[ptr % len(keys)]
            if groups[k]:
                out.append(groups[k].pop(0))
            else:
                keys.remove(k)
                if not keys:
                    break
            ptr += 1

        if len(out) < batch_size:
            out += list(reversed(buf))[: (batch_size - len(out))]
        return out[:batch_size]

    def _verify_policy(self, old_policy: str, new_policy: str, batch: List[Dict[str, Any]]) -> bool:
        # optional extra check (keep as fail-open)
        try:
            payload = {
                "old_policy": old_policy,
                "new_policy": new_policy,
                "batch": batch[: min(10, len(batch))],
                "question": "Which policy is more likely to maximize best-of-run and reduce failures? Reply JSON {\"better\":\"old\"|\"new\",\"reason\":\"...\"}."
            }
            prompt = "Evaluate routing policies. Reply STRICT JSON only.\n" + json.dumps(payload, ensure_ascii=False)
            out = self.router_llm.draw_sample(prompt)
            obj = safe_json_loads(out)
            if obj and obj.get("better") in ("old", "new"):
                return obj["better"] == "new"
        except Exception:
            pass
        return True

    def _do_textgrad_update(self, batch: List[Dict[str, Any]]):
        tg_prompt = build_textgrad_prompt(self.profiles, self.policy_text, batch, max_policy_chars=self.max_policy_chars)
        out = self.critic_llm.draw_sample(tg_prompt)
        obj = safe_json_loads(out)

        self._updates += 1

        if not obj or "new_policy" not in obj:
            self._history.append({
                "time": time.time(),
                "ok": False,
                "policy_version_before": self.policy_version,
                "error": "critic_output_not_json",
                "raw": (out or "")[:600],
            })
            return

        new_policy = str(obj.get("new_policy", "")).strip()
        rationale = str(obj.get("rationale", "")).strip()
        changes = obj.get("changes", [])

        if not policy_guardrail_ok(new_policy, self.profile_mode, self.min_policy_chars, self.max_policy_chars):
        
            # inside _do_textgrad_update(), when guardrail fails
            if self.auto_save_path:
                base = os.path.dirname(self.auto_save_path)
                fail_dir = os.path.join(base, "failed_updates")
                os.makedirs(fail_dir, exist_ok=True)
                with open(os.path.join(fail_dir, f"update_{self._updates:03d}_raw.txt"), "w", encoding="utf-8") as f:
                    f.write(out or "")
                if obj:
                    with open(os.path.join(fail_dir, f"update_{self._updates:03d}_parsed.json"), "w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False, indent=2)

            self._history.append({
                "time": time.time(),
                "ok": False,
                "policy_version_before": self.policy_version,
                "error": "policy_guardrail_failed",
                "raw": (out or "")[:600],
            })
            return

        old_policy = self.policy_text

        if self.verify_update and not self._verify_policy(old_policy, new_policy, batch):
            self._history.append({
                "time": time.time(),
                "ok": False,
                "policy_version_before": self.policy_version,
                "error": "verify_rejected",
                "rationale": rationale[:400],
            })
            return

        self.policy_text = new_policy
        self.policy_version += 1
        self._history.append({
            "time": time.time(),
            "ok": True,
            "policy_version_after": self.policy_version,
            "rationale": rationale[:600],
            "changes": changes if isinstance(changes, list) else [str(changes)],
            "batch_size": len(batch),
            "old_policy_sha256": hashlib.sha256(old_policy.encode("utf-8")).hexdigest(),
            "new_policy_sha256": hashlib.sha256(new_policy.encode("utf-8")).hexdigest(),
        })

        if self.verbose:
            print(f"[TextGradRouterV3] policy updated -> v{self.policy_version}")

        # autosave
        if self.auto_save_path and (self._updates % max(1, self.auto_save_every_updates) == 0):
            try:
                self.save_state(self.auto_save_path)
            except Exception:
                pass

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
            "verify_update": self.verify_update,

            "reward_alpha": self.reward_alpha,
            "reward_beta": self.reward_beta,
            "reward_gamma": self.reward_gamma,
            "fail_reward": self.fail_reward,
            "catastrophic_threshold": self.catastrophic_threshold,
            "catastrophic_penalty": self.catastrophic_penalty,
            "reward_shift": self.reward_shift,

            "explore_epsilon": self.explore_epsilon,
            "explore_ucb_c": self.explore_ucb_c,
            "explore_fail_weight": self.explore_fail_weight,

            "min_policy_chars": self.min_policy_chars,
            "max_policy_chars": self.max_policy_chars,

            "prior_stats": self.prior_stats.to_dict(),
            "score_stats": {k: dataclasses.asdict(v) for k, v in self.score_stats.items()},
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