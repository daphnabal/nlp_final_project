"""
Temperature schedule definitions for the final experiment.

Each function returns a list of floats of length n_chunks,
representing the sampling temperature for each story segment.
"""

from typing import List


# ---------------------------------------------------------------------------
# Final experiment schedules (N_CHUNKS=7, temperature range [0.01, 2.0])
# ---------------------------------------------------------------------------

# The 7 evenly-spaced fixed-temperature baselines used in the final experiment.
# 0.01 replaces 0.001 to avoid fp16 overflow (logits would be scaled ×100 vs ×1000).
FINAL_FIXED_TEMPS: List[float] = [0.01, 0.334, 0.667, 1.0, 1.334, 1.667, 2.0]


def _linspace(start: float, end: float, n: int) -> List[float]:
    """Return n evenly-spaced values from start to end (both inclusive)."""
    if n == 1:
        return [round(start, 4)]
    step = (end - start) / (n - 1)
    return [round(start + i * step, 4) for i in range(n)]


def get_fixed_temperature_schedules(n_chunks: int = 7) -> dict:
    """
    Fixed-temperature baselines for the final experiment.

    Each schedule holds one temperature constant across all chunks.
    The 7 values are evenly spaced on (0, 2], covering the full range from
    near-deterministic (0.01 ≈ greedy) to highly stochastic (2.0).

    Returns:
        {"fixed_temperature_0.01": [0.01]*7, ..., "fixed_temperature_2.0": [2.0]*7}
    """
    return {f"fixed_temperature_{t}": [t] * n_chunks for t in FINAL_FIXED_TEMPS}



def get_phase2_schedules(n_chunks: int = 7) -> dict:
    """
    Phase-2 dynamic schedules — capped at 1.334 (safe quality threshold).
    All schedules start and end at known fixed-temperature baselines
    {0.01, 0.334, 0.667, 1.0, 1.334}, making comparisons interpretable.

    increasing_safe  [0.01 → 1.334]:  capped increasing, anchored at known fixed temps
    decreasing_safe  [1.334 → 0.01]:  capped decreasing, anchored at known fixed temps
    peak_safe        [0.334 → 1.334 → 0.334]:  creative climax, warm structured framing
    valley_safe      [1.334 → 0.334 → 1.334]:  focused pivot, creative bookends
    """
    mid = n_chunks // 2

    valley_temps, peak_temps = [], []
    for i in range(n_chunks):
        dist = abs(i - mid) / mid if mid > 0 else 0.0
        valley_temps.append(round(0.334 + (1.334 - 0.334) * dist, 4))
        peak_temps.append(round(1.334 - (1.334 - 0.334) * dist, 4))

    return {
        "increasing_safe": _linspace(0.01, 1.334, n_chunks),
        "decreasing_safe": _linspace(1.334, 0.01, n_chunks),
        "peak_safe":       peak_temps,
        "valley_safe":     valley_temps,
    }


def get_all_final_schedules(n_chunks: int = 7) -> dict:
    """All final-experiment schedules: 7 fixed-temperature + 4 dynamic (phase 2) = 11 total."""
    return {
        **get_fixed_temperature_schedules(n_chunks),
        **get_phase2_schedules(n_chunks),
    }


def get_final_schedule(name: str, n_chunks: int = 7) -> List[float]:
    """Look up a final-experiment schedule by name."""
    all_scheds = get_all_final_schedules(n_chunks)
    if name not in all_scheds:
        raise ValueError(
            f"Unknown schedule '{name}'. "
            f"Options: {list(all_scheds.keys())}"
        )
    return all_scheds[name]
