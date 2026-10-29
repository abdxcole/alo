"""Efectos de sonido estilo Dragon Ball sintetizados a partir de lo que pasa en cada cuadro.

Capas: rugido del aura, zumbido de ki que sube de tono con la carga, chispazos eléctricos por cada
rayo dibujado, impacto en la fusión, carga creciente antes del ataque, explosión y rugido del haz.
En la cámara lenta los tonos bajan, como en el cine."""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, sosfilt

SR = 48000


def _band(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)


def _low(x, f, order=2):
    return sosfilt(butter(order, f, btype="low", fs=SR, output="sos"), x)


def _high(x, f, order=2):
    return sosfilt(butter(order, f, btype="high", fs=SR, output="sos"), x)


def _norm(x):
    return x / (np.max(np.abs(x)) + 1e-9)


def _env(log, key, n, fps):
    v = np.array([f[key] for f in log], np.float64)
    return np.interp(np.arange(n) / SR, np.arange(len(v)) / fps, v)


def _events(log, key, thr):
    """Cuadros donde la señal cruza el umbral hacia arriba."""
    v = np.array([f[key] for f in log])
    return [i for i in range(1, len(v)) if v[i] >= thr > v[i - 1]]


def _boom(dur, f0, f1, rng, pitch=1.0):
    t = np.arange(int(dur * SR)) / SR
    f = (f1 + (f0 - f1) * np.exp(-t / 0.12)) * pitch
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (dur * 0.45))
    body = _low(rng.normal(size=len(t)), 1800) * np.exp(-t / 0.18)
    crack = _high(rng.normal(size=len(t)), 2500) * np.exp(-t / 0.012)
    return np.tanh(1.6 * (sub * 1.0 + _norm(body) * 0.55 + _norm(crack) * 0.35))


def _zap(rng, pitch=1.0):
    dur = rng.uniform(0.02, 0.07)
    t = np.arange(int(dur * SR)) / SR
    buzz = np.sign(np.sin(2 * np.pi * rng.uniform(70, 160) * pitch * t))
    z = _high(rng.normal(size=len(t)), 1800) * (0.6 + 0.4 * buzz) * np.exp(-t / rng.uniform(0.006, 0.02))
    return _norm(z) * rng.uniform(0.3, 1.0)


def synthesize(log, fps, path, seed=5):
    rng = np.random.default_rng(seed)
    n = int(len(log) / fps * SR) + SR // 2
    t = np.arange(n) / SR
    aura = _env(log, "aura", n, fps)
    orb = np.clip(_env(log, "orb", n, fps), 0, 2)
    beam = _env(log, "beam", n, fps)
    rate = np.clip(_env(log, "rate", n, fps), 0.2, 1.5)
    pitch = rate ** 0.35  # cámara lenta = tonos más graves
    out = np.zeros((n, 2))

    # 1) aura: viento/llama con aleteo + retumbo grave
    flutter = _norm(_low(rng.normal(size=n), 18))
    for ch in range(2):
        roar = _norm(_band(rng.normal(size=n), 140, 1100)) * (1 + 0.35 * flutter)
        out[:, ch] += roar * aura * 0.26
    rumble = _norm(_low(rng.normal(size=n), 75, 4))
    out += (rumble * aura * 0.45)[:, None]

    # 2) ki: zumbido eléctrico (diente de sierra) + silbido que sube de tono con la carga
    orb_s = _low(orb, 3)
    f0 = (85 + 45 * orb_s) * pitch * (1 + 0.015 * np.sin(2 * np.pi * 6 * t))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    saw = sum(np.sin(k * ph) / k for k in range(1, 12))
    hum = np.tanh(2.0 * _band(saw, 150, 3500))
    whine_f = (700 + 900 * np.clip(orb_s, 0, 1.6)) * pitch
    whine = np.sin(2 * np.pi * np.cumsum(whine_f) / SR) * (0.75 + 0.25 * np.sin(2 * np.pi * 23 * t))
    out += ((hum * 0.16 + whine * 0.05) * np.clip(orb_s, 0, 1.5))[:, None]

    # 3) chispazos: uno por cada rayo dibujado, repartidos dentro del cuadro
    for j, f in enumerate(log):
        # no todos los rayos suenan: un chispazo por rayo sería un siseo constante
        for _ in range(rng.binomial(f["bolts"], 0.3)):
            z = _zap(rng, pitch[min(int(j / fps * SR), n - 1)])
            i0 = int((j + rng.uniform(0, 1)) / fps * SR)
            i1 = min(n, i0 + len(z))
            pan = rng.uniform(0.25, 0.75)
            out[i0:i1, 0] += z[:i1 - i0] * 0.16 * (1 - pan)
            out[i0:i1, 1] += z[:i1 - i0] * 0.16 * pan

    # 4) fusión: impacto
    for j in _events(log, "burst", 0.5):
        p = pitch[int(j / fps * SR)]
        b = _boom(1.4, 140, 45, rng, p)
        i0 = int(j / fps * SR)
        i1 = min(n, i0 + len(b))
        out[i0:i1] += (b[:i1 - i0] * 0.55)[:, None]

    # 5) lanzamiento: carga creciente antes, explosión, rugido del haz y cola
    for j in _events(log, "beam", 0.05):
        i0 = int(j / fps * SR)
        p = pitch[i0]
        pre = int(0.9 * SR)
        a = max(0, i0 - pre)
        k = (np.arange(i0 - a) / pre)[-(i0 - a):] if i0 > a else np.zeros(0)
        swell_f = (300 + 2200 * k ** 2) * p
        swell = np.sin(2 * np.pi * np.cumsum(swell_f) / SR) * 0.3 + _norm(_band(rng.normal(size=len(k)), 400, 5000)) * 0.5
        out[a:i0] += (swell * k ** 2.5 * 0.5)[:, None]
        b = _boom(2.5, 120, 32, rng, p)
        i1 = min(n, i0 + len(b))
        out[i0:i1] += (b[:i1 - i0] * 0.75)[:, None]
    beam_roar = _norm(_band(rng.normal(size=n), 180, 4500)) * (1 + 0.4 * _norm(_low(rng.normal(size=n), 35)))
    ph = 2 * np.pi * np.cumsum(55 * pitch) / SR
    beam_tone = np.tanh(3 * sum(np.sin(k * ph) / k for k in range(1, 8)))
    out += ((beam_roar * 0.45 + beam_tone * 0.2) * beam)[:, None]

    # reverberación corta de cuarto + limitador suave
    ir_t = np.arange(int(1.1 * SR)) / SR
    ir = _low(rng.normal(size=len(ir_t)), 5000) * np.exp(-ir_t / 0.35)
    ir /= np.sqrt((ir ** 2).sum())
    wet = np.stack([fftconvolve(out[:, c], ir)[:n] for c in range(2)], 1)
    mix = out * 0.8 + wet * 0.35
    mix = _high(mix.T, 25).T
    mix = mix / (np.max(np.abs(mix)) + 1e-9)
    mix = np.tanh(1.8 * mix) / np.tanh(1.8) * 0.89
    wavfile.write(path, SR, (mix * 32767).astype(np.int16))
    return path
