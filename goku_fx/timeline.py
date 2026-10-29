"""Detecta las fases del ataque (carga, fusión, lanzamiento) a partir del movimiento de las manos."""

import json

import numpy as np

from .track import _gauss1d, hand_geometry


def smoothstep(e0, e1, x):
    d = e1 - e0
    x = np.clip((np.asarray(x, np.float32) - e0) / (d if abs(d) > 1e-6 else 1e-6), 0, 1)
    return x * x * (3 - 2 * x)


def detect(lm, alpha, fps, overrides=None):
    """Devuelve los eventos en segundos: charge_start, fuse, launch, launch_dir, mode, beam_duration."""
    ov = dict(overrides or {})
    n = lm.shape[1]
    center, scale = hand_geometry(lm)  # (2, n, 2), (2, n)
    vis = alpha > 0.5
    both = vis[0] & vis[1]
    any_ = vis[0] | vis[1]
    sc = np.where(both, scale.mean(0), np.where(vis[0], scale[0], scale[1]))
    sc = np.maximum(sc, 1.0)
    d_norm = np.linalg.norm(center[0] - center[1], axis=-1) / sc

    ev = {}
    first = int(np.argmax(any_)) if any_.any() else 0
    ev["charge_start"] = ov.get("charge_start", first / fps)

    # fusión: ambas manos juntas al menos ~0.1 s
    fuse = None
    need = max(2, int(0.1 * fps))
    run = 0
    for t in range(int(ev["charge_start"] * fps), n):
        run = run + 1 if (both[t] and d_norm[t] < 2.1) else 0
        if run >= need:
            fuse = (t - need + 1) / fps
            break
    ev["fuse"] = ov.get("fuse", fuse)

    # lanzamiento: empuje brusco (velocidad en pantalla o crecimiento hacia la cámara)
    joint = np.where(both[:, None], center.mean(0), np.where(vis[0][:, None], center[0], center[1]))
    joint = _gauss1d(joint.astype(np.float32), 1.0)
    # la mano más grande (más cerca de la cámara) delata el empujón hacia el frente
    big = np.maximum(np.where(vis[0], scale[0], 0), np.where(vis[1], scale[1], 0))
    big = np.where(any_, big, sc)
    med = np.array([np.median(big[max(0, t - 2):t + 3]) for t in range(n)], np.float32)  # quita saltos de escala
    s_s = _gauss1d(med[:, None], 1.0)[:, 0]
    vel = np.gradient(joint, axis=0)
    growth = np.gradient(s_s) / s_s
    speed = np.linalg.norm(vel, axis=1) / s_s
    push = _gauss1d((speed + 4.0 * np.maximum(growth, 0))[:, None], 1.0)[:, 0] * any_
    start = ev["fuse"] if ev["fuse"] is not None else ev["charge_start"] + 1.0
    s0 = min(int((start + 0.35) * fps), n - 1)
    launch, direction, mode = None, [1.0, 0.0], "beam"
    if s0 < n - 2:
        seg = push[s0:]
        top = float(seg.max())
        # el primer empujón fuerte es el ataque; lo que viene después suele ser bajar los brazos
        peak = next(k for k in range(len(seg)) if seg[k] >= 0.5 * top and (k + 1 == len(seg) or seg[k + 1] < seg[k]))
        if top > 0.09:
            on = peak
            while on > 0 and seg[on - 1] > 0.45 * seg[peak]:
                on -= 1
            lf = s0 + on
            launch = lf / fps
            w = slice(lf, min(n, s0 + peak + int(0.12 * fps) + 1))
            v = vel[w].sum(0) / s_s[lf]
            g = growth[w].sum()
            if g > 0.22 or np.linalg.norm(v) < 0.4:
                mode = "camera"
            else:
                direction = (v / np.linalg.norm(v)).tolist()
    ev["launch"] = ov.get("launch", launch)
    ev["launch_dir"] = ov.get("launch_dir", direction)
    ev["mode"] = ov.get("mode", mode)
    ev["beam_duration"] = ov.get("beam_duration", 0.9 if ev["mode"] == "camera" else 1.6)
    ev["aura"] = ov.get("aura", True)
    return ev


def save(ev, path):
    with open(path, "w") as f:
        json.dump(ev, f, indent=2, ensure_ascii=False)


def load(path):
    with open(path) as f:
        return json.load(f)


def describe(ev):
    f = lambda x: "—" if x is None else f"{x:.2f}s"
    d = ev["launch_dir"]
    where = "hacia la cámara" if ev["mode"] == "camera" else f"dirección ({d[0]:+.2f}, {d[1]:+.2f})"
    return (f"carga {f(ev['charge_start'])} · fusión {f(ev['fuse'])} · "
            f"lanzamiento {f(ev['launch'])} {where if ev['launch'] is not None else ''}")
