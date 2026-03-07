"""
Temperature schedule definitions.

Each function returns a list of floats of length n_chunks,
representing the sampling temperature for each story segment.
"""

from typing import List


def fixed(n_chunks: int = 3, tau: float = 0.7) -> List[float]:
    """Baseline: constant temperature across all chunks."""
    return [tau] * n_chunks


def increasing(n_chunks: int = 3, low: float = 0.5, high: float = 1.3) -> List[float]:
    """Start cool (structured), warm up to creative."""
    if n_chunks == 1:
        return [low]
    step = (high - low) / (n_chunks - 1)
    return [round(low + i * step, 4) for i in range(n_chunks)]


def decreasing(n_chunks: int = 3, high: float = 1.3, low: float = 0.5) -> List[float]:
    """Start creative, cool down to coherent ending."""
    if n_chunks == 1:
        return [high]
    step = (high - low) / (n_chunks - 1)
    return [round(high - i * step, 4) for i in range(n_chunks)]


def valley(n_chunks: int = 3, low: float = 0.5, high: float = 1.3) -> List[float]:
    """High-low-high: creative opening, focused middle, creative ending."""
    if n_chunks == 1:
        return [high]
    if n_chunks == 2:
        return [high, low]
    mid = n_chunks // 2
    temps = []
    for i in range(n_chunks):
        # cosine-like: peaks at edges, trough at center
        dist_from_center = abs(i - mid) / mid  # 0 at center, 1 at edges
        t = low + (high - low) * dist_from_center
        temps.append(round(t, 4))
    return temps


def peak(n_chunks: int = 3, low: float = 0.5, high: float = 1.3) -> List[float]:
    """Low-high-low: structured opening, creative middle, structured ending."""
    if n_chunks == 1:
        return [low]
    if n_chunks == 2:
        return [low, high]
    mid = n_chunks // 2
    temps = []
    for i in range(n_chunks):
        dist_from_center = abs(i - mid) / mid
        t = high - (high - low) * dist_from_center
        temps.append(round(t, 4))
    return temps


def get_all_schedules(n_chunks: int = 3) -> dict:
    """Return the 5 main named schedules as {name: [temps]} dict."""
    return {
        "fixed":      fixed(n_chunks),
        "increasing": increasing(n_chunks),
        "decreasing": decreasing(n_chunks),
        "valley":     valley(n_chunks),
        "peak":       peak(n_chunks),
    }


def get_sweep_schedules(temps: List[float], n_chunks: int = 3) -> dict:
    """
    Return fixed-temperature sweep schedules as {sweep_T: [temps]} dict.

    E.g. get_sweep_schedules([0.5, 0.7, 1.0, 1.2, 1.5]) →
         {"sweep_0.5": [0.5,0.5,0.5], "sweep_0.7": [0.7,0.7,0.7], ...}
    """
    return {f"sweep_{t}": fixed(n_chunks, t) for t in temps}


def get_schedule(name: str, n_chunks: int = 3) -> List[float]:
    """
    Look up a schedule by name.

    Handles both named schedules (fixed, increasing, ...) and
    sweep schedules of the form 'sweep_<tau>' (e.g. 'sweep_1.2').
    """
    if name.startswith("sweep_"):
        try:
            tau = float(name[len("sweep_"):])
        except ValueError:
            raise ValueError(f"Invalid sweep schedule name '{name}'. Expected 'sweep_<float>'.")
        return fixed(n_chunks, tau)
    schedules = get_all_schedules(n_chunks)
    if name not in schedules:
        raise ValueError(f"Unknown schedule '{name}'. Options: {list(schedules.keys())} or 'sweep_<tau>'.")
    return schedules[name]
