from __future__ import annotations

import time


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
