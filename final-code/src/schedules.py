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


def get_final_named_schedules(n_chunks: int = 7) -> dict:
    """
    Named dynamic temperature schedules for the final experiment.

    All schedules span the same [0.01, 2.0] grid as the fixed-temperature
    baselines, making comparisons maximally interpretable.  For n_chunks=7
    the valley/peak patterns use only every-other grid point
    {0.01, 0.667, 1.334, 2.0}, producing sharper contrasts.

    increasing  [0.01 → 2.0]:
        Structured world-building early anchors narrative context; the model
        has strong prior context by mid-story so late high-T tokens are
        "guided" by coherent text already generated.  Tests whether narrative
        bootstrapping reduces the coherence cost of high later temperatures.

    decreasing  [2.0 → 0.01]:
        Analogous to simulated annealing: divergent creative exploration first,
        then progressive cooling toward a precise, coherent resolution.  Tests
        whether creative sparks early can be converged into a coherent ending.

    valley  [high → low → high]:
        Near-greedy at chunk 4 (the narrative pivot / "turn"), creative at the
        bookends.  The midpoint is structurally critical, so maximum precision
        there minimises coherence loss at the most load-bearing plot junction.

    peak  [low → high → low]:
        Structured opening establishes the story world; structured close ensures
        satisfying resolution; maximum creativity at the climax (chunk 4).
        Tests the "creative heart" hypothesis: coherence is mainly determined
        by structural framing, not the middle segment.
    """
    high, low = 2.0, 0.01
    mid = n_chunks // 2

    valley_temps, peak_temps = [], []
    for i in range(n_chunks):
        dist = abs(i - mid) / mid if mid > 0 else 0.0   # 0 at center, 1 at edges
        valley_temps.append(round(low + (high - low) * dist, 4))
        peak_temps.append(round(high - (high - low) * dist, 4))

    return {
        "increasing": _linspace(low, high, n_chunks),
        "decreasing": _linspace(high, low, n_chunks),
        "valley":     valley_temps,
        "peak":       peak_temps,
    }


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
    """All final-experiment schedules: 7 fixed-temperature + 4 named dynamic = 11 total."""
    return {
        **get_fixed_temperature_schedules(n_chunks),
        **get_final_named_schedules(n_chunks),
    }


def get_final_schedule(name: str, n_chunks: int = 7) -> List[float]:
    """Look up a final-experiment schedule by name (phase 1 or phase 2)."""
    all_scheds = {**get_all_final_schedules(n_chunks), **get_phase2_schedules(n_chunks)}
    if name not in all_scheds:
        raise ValueError(
            f"Unknown final schedule '{name}'. "
            f"Options: {list(all_scheds.keys())}"
        )
    return all_scheds[name]
