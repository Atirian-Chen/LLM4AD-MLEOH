# router_trace.py
from __future__ import annotations

import os
import csv
import threading
from typing import Optional


class RouterTracer:
    """
    Append-only CSV tracer for router decisions.

    Each row:
        seed, step, operator, chosen_model

    - Thread-safe: protected by a lock.
    - Low overhead: open file in append mode per log (safe across processes too).
    """

    def __init__(self, trace_path: str):
        self.trace_path = str(trace_path)
        self.step = 0
        self._lock = threading.Lock()

        # ensure directory
        d = os.path.dirname(self.trace_path)
        if d:
            os.makedirs(d, exist_ok=True)

        # create file with header if not exists
        if not os.path.exists(self.trace_path):
            with open(self.trace_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["seed", "step", "operator", "chosen_model"])

    def log(self, seed: int, operator: Optional[str], chosen_model: str) -> None:
        op = (operator or "unknown").upper()
        chosen = str(chosen_model).lower()

        with self._lock:
            self.step += 1
            with open(self.trace_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([int(seed), int(self.step), op, chosen])
