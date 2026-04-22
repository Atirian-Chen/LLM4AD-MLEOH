# bandit_router.py
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =============================
# 1) Core data structure
# =============================
@dataclass
class Decision:
    arm: int
    x: List[float]
    operator: str
    chosen_model: str


# =============================
# 2) Feature extraction
# =============================
_OP_RE = re.compile(r"(operator|op)\s*[:=]\s*(E1|E2|M1|M2)\b", flags=re.IGNORECASE)
_OP_ANY_RE = re.compile(r"\b(E1|E2|M1|M2)\b")


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


def featurize(prompt: Any) -> Tuple[List[float], str]:
    """
    返回 (x, operator)
    x 用于 LinUCB 的上下文特征。这里保持“轻量、鲁棒、可解释”的特征集合。

    x = [
      1,                       # bias
      onehot(E1,E2,M1,M2,UNK),  # 5
      log(1+chars),
      log(1+lines),
      code_density,
      has_error,
      has_fix_signal,
      has_explore_signal,
    ]
    """
    text = _prompt_to_text(prompt)
    tl = text.lower()

    op = _detect_operator(text)
    op_list = ["E1", "E2", "M1", "M2", "UNKNOWN"]
    op_onehot = [1.0 if op == o else 0.0 for o in op_list]

    n_chars = len(text)
    n_lines = len(text.splitlines())

    # 粗略代码密度（不会太敏感）
    code_markers = 0
    for pat in ["def ", "class ", "```", "return ", " for ", " while ", "import ", "np.", "numpy"]:
        code_markers += tl.count(pat)
    code_density = min(1.0, code_markers / 25.0)

    error_kw = [
        "traceback", "exception", "syntaxerror", "typeerror", "nameerror",
        "valueerror", "indexerror", "keyerror", "assertionerror", "indentationerror",
    ]
    has_error = 1.0 if any(k in tl for k in error_kw) else 0.0

    fix_words = ["fix", "bug", "repair", "patch", "debug", "crash", "error", "exception", "modify", "edit", "refine"]
    explore_words = ["design", "propose", "invent", "rewrite", "new heuristic", "explore", "diversify", "variant"]
    has_fix_signal = 1.0 if any(w in tl for w in fix_words) else 0.0
    has_explore_signal = 1.0 if any(w in tl for w in explore_words) else 0.0

    x = [
        1.0,
        *op_onehot,
        math.log1p(n_chars),
        math.log1p(n_lines),
        code_density,
        has_error,
        has_fix_signal,
        has_explore_signal,
    ]
    return x, op


# =============================
# 3) LinUCB (supports n_arms)
# =============================
class LinUCB:
    def __init__(self, dim: int, alpha: float = 1.0, l2: float = 1.0, eps: float = 0.05, n_arms: int = 2):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.eps = float(eps)
        self.n_arms = int(n_arms)

        self._lock = threading.Lock()
        self.A = [self.l2 * np.eye(self.dim, dtype=np.float64) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.dim, dtype=np.float64) for _ in range(self.n_arms)]

    def select(self, x: np.ndarray) -> int:
        with self._lock:
            # epsilon-greedy exploration
            if np.random.rand() < self.eps:
                return int(np.random.randint(0, self.n_arms))

            best_arm = 0
            best_p = -1e30
            for a in range(self.n_arms):
                A_inv = np.linalg.inv(self.A[a])
                theta = A_inv @ self.b[a]
                mean = float(theta @ x)
                bonus = float(self.alpha * math.sqrt(max(0.0, x @ A_inv @ x)))
                p = mean + bonus
                if p > best_p:
                    best_p = p
                    best_arm = a
            return int(best_arm)

    def update(self, arm: int, x: np.ndarray, r: float) -> None:
        arm = int(arm)
        with self._lock:
            self.A[arm] += np.outer(x, x)
            self.b[arm] += float(r) * x

    def state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "dim": self.dim,
                "alpha": self.alpha,
                "l2": self.l2,
                "eps": self.eps,
                "n_arms": self.n_arms,
                "A": [a.tolist() for a in self.A],
                "b": [b.tolist() for b in self.b],
            }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        with self._lock:
            self.dim = int(d["dim"])
            self.alpha = float(d["alpha"])
            self.l2 = float(d["l2"])
            self.eps = float(d["eps"])
            self.n_arms = int(d.get("n_arms", 2))
            self.A = [np.array(a, dtype=np.float64) for a in d["A"]]
            self.b = [np.array(b, dtype=np.float64) for b in d["b"]]


# =============================
# 4) Bandit Router LLM (3 backends)
# =============================
class BanditRouterLLM:
    """
    3 LLM 版本：
      arm 0 -> deepseek
      arm 1 -> qwen
      arm 2 -> doubao
    """

    def __init__(
        self,
        deepseek_llm,
        qwen_llm,
        doubao_llm,
        bandit: LinUCB,
        verbose: bool = False,
        tracer=None,     # RouterTracer（可选）
        seed_id: int = -1,
    ):
        self.deepseek_llm = deepseek_llm
        self.qwen_llm = qwen_llm
        self.doubao_llm = doubao_llm
        self.bandit = bandit
        self.verbose = verbose

        self.tracer = tracer
        self.seed_id = int(seed_id)

        self.arm_names = ["deepseek", "qwen", "doubao"]

        # thread-local last decision
        self._tl = threading.local()

        # stats
        self.call_cnt = 0
        self.deepseek_cnt = 0
        self.qwen_cnt = 0
        self.doubao_cnt = 0

        # sanity
        if self.bandit.n_arms != 3:
            # 允许你误传，自动修正到 3（避免 silently wrong）
            self.bandit.n_arms = 3
            # 但 A/b 维度也要对齐
            if len(self.bandit.A) != 3 or len(self.bandit.b) != 3:
                self.bandit.A = [self.bandit.l2 * np.eye(self.bandit.dim) for _ in range(3)]
                self.bandit.b = [np.zeros(self.bandit.dim) for _ in range(3)]

    def set_seed_id(self, seed: int) -> None:
        self.seed_id = int(seed)

    def _set_last_decision(self, d: Decision) -> None:
        self._tl.last_decision = d

    def pop_last_decision(self) -> Optional[Decision]:
        d = getattr(self._tl, "last_decision", None)
        if hasattr(self._tl, "last_decision"):
            delattr(self._tl, "last_decision")
        return d

    def _pick_backend(self, arm: int):
        if arm == 0:
            self.deepseek_cnt += 1
            return self.deepseek_llm
        if arm == 1:
            self.qwen_cnt += 1
            return self.qwen_llm
        self.doubao_cnt += 1
        return self.doubao_llm

    def draw_sample(self, prompt: Any, *args, **kwargs) -> Any:
        self.call_cnt += 1

        x_list, op = featurize(prompt)
        x = np.array(x_list, dtype=np.float64)

        arm = int(self.bandit.select(x))
        chosen_model = self.arm_names[arm]
        backend = self._pick_backend(arm)

        # save decision for BanditEoH.update(...)
        self._set_last_decision(Decision(arm=arm, x=[float(v) for v in x_list], operator=op, chosen_model=chosen_model))

        # trace (optional)
        if self.tracer is not None:
            try:
                self.tracer.log(self.seed_id, op, chosen_model)
            except Exception:
                pass

        if self.verbose:
            print(f"[BanditRouter] choose={chosen_model} op={op} seed={self.seed_id}")

        return backend.draw_sample(prompt, *args, **kwargs)

    def draw_samples(self, *args, **kwargs):
        n = kwargs.pop("n", None)
        if n is not None:
            n = int(n)
            return [self.draw_sample(*args, **kwargs) for _ in range(n)]
        return self.draw_sample(*args, **kwargs)

    def save_bandit(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.bandit.state_dict(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_bandit(path: str) -> LinUCB:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        bandit = LinUCB(
            dim=int(d["dim"]),
            alpha=float(d.get("alpha", 1.0)),
            l2=float(d.get("l2", 1.0)),
            eps=float(d.get("eps", 0.05)),
            n_arms=int(d.get("n_arms", 3)),
        )
        bandit.load_state_dict(d)
        return bandit

    def close(self):
        # best-effort close
        for b in (self.deepseek_llm, self.qwen_llm, self.doubao_llm):
            if hasattr(b, "close"):
                try:
                    b.close()
                except Exception:
                    pass


# ==========================================================
# 5) Your original plotting CLI (updated to 3 models)
# ==========================================================
def load_trace(trace_path: str):
    import pandas as pd  # lazy import
    df = pd.read_csv(trace_path)
    df["chosen_model"] = df["chosen_model"].astype(str).str.lower().str.strip()
    df["operator"] = df["operator"].astype(str).str.upper().str.strip()

    # 兜底：支持三模型
    df.loc[~df["chosen_model"].isin(["deepseek", "qwen", "doubao"]), "chosen_model"] = "unknown"
    df.loc[df["operator"].isin(["NONE", "NAN", ""]), "operator"] = "UNKNOWN"
    return df


def plot_operator_stacked(df, out_path: str):
    import numpy as np
    import matplotlib.pyplot as plt

    g = df.groupby(["operator", "chosen_model"]).size().reset_index(name="count")
    pivot = g.pivot_table(index="operator", columns="chosen_model", values="count", fill_value=0)

    for col in ["deepseek", "qwen", "doubao"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["deepseek", "qwen", "doubao"]]

    prop = pivot.div(pivot.sum(axis=1), axis=0)

    plt.figure(figsize=(8, 4))
    bottom = np.zeros(len(prop))
    for col in prop.columns:
        plt.bar(prop.index, prop[col].values, bottom=bottom, label=col)
        bottom += prop[col].values

    plt.ylim(0, 1)
    plt.ylabel("Choice Proportion")
    plt.title("Bandit Router: Choice Distribution by Operator")
    plt.xticks(rotation=20)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[OK] saved: {out_path}")


def plot_cumulative_curve(df, out_path: str):
    """
    3 模型累计比例曲线（更适合你的 3-LLM 实验）
    """
    import numpy as np
    import matplotlib.pyplot as plt

    df2 = df.sort_values("step").reset_index(drop=True)
    steps = np.arange(1, len(df2) + 1)

    plt.figure(figsize=(8, 4))
    for name in ["deepseek", "qwen", "doubao"]:
        is_m = (df2["chosen_model"] == name).astype(int).values
        cum = np.cumsum(is_m) / steps
        plt.plot(steps, cum, label=f"Cumulative P({name})")

    plt.ylim(0, 1)
    plt.xlabel("Routing Step")
    plt.ylabel("Cumulative Choice Probability")
    plt.title("Bandit Router: Cumulative Preference Over Time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[OK] saved: {out_path}")


def plot_heatmap(df, out_path: str):
    import numpy as np
    import matplotlib.pyplot as plt

    g = df.groupby(["operator", "chosen_model"]).size().reset_index(name="count")
    pivot = g.pivot_table(index="operator", columns="chosen_model", values="count", fill_value=0)

    for col in ["deepseek", "qwen", "doubao"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["deepseek", "qwen", "doubao"]]
    prop = pivot.div(pivot.sum(axis=1), axis=0)

    mat = prop.values
    ops = prop.index.tolist()
    models = prop.columns.tolist()

    plt.figure(figsize=(7, 4))
    plt.imshow(mat, aspect="auto")
    plt.xticks(range(len(models)), models)
    plt.yticks(range(len(ops)), ops)
    plt.colorbar(label="Choice Proportion")
    plt.title("Bandit Router: Operator-Model Preference Heatmap")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[OK] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=str, required=True, help="Path to router_trace.csv")
    parser.add_argument("--out_dir", type=str, default=".", help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_trace(args.trace)

    plot_operator_stacked(df, os.path.join(args.out_dir, "fig_bandit_choice_by_operator.png"))
    plot_cumulative_curve(df, os.path.join(args.out_dir, "fig_bandit_choice_cumulative.png"))
    plot_heatmap(df, os.path.join(args.out_dir, "fig_bandit_choice_heatmap.png"))


if __name__ == "__main__":
    main()