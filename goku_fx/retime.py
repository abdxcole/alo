"""Cámara lenta: rampa de velocidad que frena en la fusión y el lanzamiento, como en una película."""

import numpy as np

from .fx import smoothstep


def speed_at(s, ev, mode):
    if mode == "no":
        return 1.0
    if mode != "rampa":
        return float(mode)
    base, slow = 0.75, 0.33
    fu, la = ev["fuse"], ev["launch"]
    if fu is None and la is None:
        return 0.6
    a = (fu if fu is not None else la - 0.6) - 0.35
    b = (la if la is not None else fu + 1.0) + 0.6
    k = smoothstep(a - 0.3, a, s) * (1 - smoothstep(b, b + 0.4, s))
    return base + (slow - base) * float(k)


def build(ev, duration, out_fps, mode="rampa"):
    """τ (segundos del video original) de cada cuadro de salida."""
    ts, s = [], 0.0
    while s < duration - 1e-6:
        ts.append(s)
        s += speed_at(s, ev, mode) / out_fps
    return np.array(ts)
