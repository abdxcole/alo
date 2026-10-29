"""Primitivas de efectos en luz lineal HDR: plasma, rayos, partículas, haz, aura, bloom y tonemap.

Todas las capas son float32 BGR a resolución de efectos (la mitad del video); los valores > 1
son luz "sobreexpuesta" que el tonemap convierte en núcleo blanco, como en una cámara real."""

import cv2
import numpy as np

TAU = 2 * np.pi

PALETTES = {  # RGB lineal
    "kamehameha": dict(core=(0.92, 0.97, 1.0), mid=(0.22, 0.62, 1.0), fil=(0.6, 0.88, 1.0), outer=(0.03, 0.18, 1.0)),
    "azul": dict(core=(0.95, 0.98, 1.0), mid=(0.45, 0.76, 1.0), fil=(0.8, 0.92, 1.0), outer=(0.1, 0.3, 1.0)),
    "dorado": dict(core=(1.0, 1.0, 0.9), mid=(1.0, 0.78, 0.25), fil=(1.0, 0.93, 0.6), outer=(1.0, 0.42, 0.04)),
    "blanco": dict(core=(1.0, 1.0, 1.0), mid=(0.85, 0.9, 1.0), fil=(1.0, 1.0, 1.0), outer=(0.45, 0.55, 1.0)),
    "morado": dict(core=(1.0, 0.95, 1.0), mid=(0.78, 0.45, 1.0), fil=(0.92, 0.75, 1.0), outer=(0.42, 0.06, 1.0)),
    "rojo": dict(core=(1.0, 0.95, 0.9), mid=(1.0, 0.35, 0.2), fil=(1.0, 0.7, 0.55), outer=(0.9, 0.05, 0.02)),
}


class Palette:
    def __init__(self, name):
        p = PALETTES[name]
        for k, v in p.items():
            setattr(self, k, np.array(v[::-1], np.float32))  # a BGR


def smoothstep(e0, e1, x):
    d = e1 - e0
    x = np.clip((x - e0) / (d if abs(d) > 1e-6 else 1e-6), 0.0, 1.0)
    return x * x * (3 - 2 * x)


class Noise:
    """Ruido fractal 1/f embaldosable (FFT); se muestrea con remap y envoltura, así es barato animarlo."""

    def __init__(self, size=512, seed=11):
        self.size = size
        self.tex = [self._make(size, b, seed + i) for i, b in enumerate((3.4, 2.8))]

    @staticmethod
    def _make(size, beta, seed):
        rng = np.random.default_rng(seed)
        f = np.sqrt(np.fft.fftfreq(size)[:, None] ** 2 + np.fft.fftfreq(size)[None, :] ** 2)
        f[0, 0] = 1
        spec = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))) / f ** (beta / 2)
        spec[0, 0] = 0
        n = np.real(np.fft.ifft2(spec))
        return ((n - n.mean()) / n.std()).astype(np.float32)

    def sample(self, u, v, which=0):
        s = self.size
        return cv2.remap(self.tex[which], np.mod(u, s).astype(np.float32), np.mod(v, s).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _roi(shape, c, ext):
    h, w = shape[:2]
    x0, x1 = int(max(0, c[0] - ext)), int(min(w, c[0] + ext + 1))
    y0, y1 = int(max(0, c[1] - ext)), int(min(h, c[1] + ext + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    X, Y = np.meshgrid(np.arange(x0, x1, dtype=np.float32) - c[0], np.arange(y0, y1, dtype=np.float32) - c[1])
    return (slice(y0, y1), slice(x0, x1)), X, Y


def orb(E, heat, c, R, t, energy, pal, noise, seed=0, hot=1.0):
    """Esfera de plasma: núcleo sobreexpuesto, cuerpo turbulento que fluye hacia fuera y filamentos."""
    if energy <= 0.003 or R < 0.5:
        return
    r_ = _roi(E.shape, c, 5.0 * R)
    if r_ is None:
        return
    sl, X, Y = r_
    s = noise.size
    r = np.sqrt(X * X + Y * Y) / R + 1e-4
    # frecuencias bajas cerca del centro para que no haya aliasing en esferas pequeñas
    u = (np.arctan2(Y, X) / TAU + 0.5) * s + seed * 97
    lr = np.log(r + 0.35)
    n1 = noise.sample(u, lr * 0.3 * s - t * 0.5 * s + seed * 31, 0)
    n2 = noise.sample(u * 2 + 13, lr * 0.5 * s - t * 1.2 * s + seed * 17, 0)
    rd = r * np.abs(1 + 0.2 * n1 + 0.08 * n2)
    core = np.exp(-(r * (1 + 0.06 * n1) / 0.3) ** 2) * (1.9 * hot)
    body = np.exp(-(rd / 0.85) ** 3) * 0.7
    fil = np.clip(1 - np.abs(n2) * 1.3, 0, 1) ** 5 * np.exp(-(r / 1.1) ** 2) * 0.9
    glow = 0.5 / (1 + (r * 2.0) ** 2)
    E[sl] += energy * (core[..., None] * pal.core + body[..., None] * pal.mid
                       + fil[..., None] * pal.fil + glow[..., None] * pal.outer)
    heat[sl] += energy * np.exp(-(r / 1.6) ** 2)


def bolt_points(p0, p1, rng, rough=0.32, depth=5):
    pts = [np.asarray(p0, np.float32), np.asarray(p1, np.float32)]
    for _ in range(depth):
        new = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            seg = b - a
            L = float(np.hypot(*seg)) + 1e-6
            m = (a + b) / 2 + np.array([-seg[1], seg[0]], np.float32) / L * rng.normal() * L * rough
            new += [m, b]
        pts = new
    return np.array(pts, np.float32)


def draw_bolt(A, p0, p1, rng, color, rough=0.32, branches=1):
    pts = bolt_points(p0, p1, rng, rough)
    cv2.polylines(A, [np.round(pts).astype(np.int32)], False, tuple(float(x) for x in color), 1, cv2.LINE_8)
    for _ in range(branches):
        if rng.random() < 0.6:
            k = rng.integers(len(pts) // 4, 3 * len(pts) // 4)
            d = pts[-1] - pts[0]
            ang = rng.normal() * 0.7
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
            end = pts[k] + rot @ d * rng.uniform(0.25, 0.5)
            draw_bolt(A, pts[k], end, rng, color * 0.6, rough, 0)


def orb_bolts(A, c, R, energy, rng, pal, rate=2.5):
    """Descargas eléctricas que saltan de la superficie de la esfera (nuevas cada cuadro, como un arco real)."""
    for _ in range(rng.poisson(rate * energy)):
        a = rng.uniform(0, TAU)
        p0 = c + 0.55 * R * np.array([np.cos(a), np.sin(a)])
        a2 = a + rng.normal() * 0.5
        p1 = c + R * rng.uniform(1.3, 2.6) * np.array([np.cos(a2), np.sin(a2)])
        draw_bolt(A, p0, p1, rng, (pal.mid * 0.6 + pal.fil * 0.4) * rng.uniform(0.7, 1.4) * energy)


class Particles:
    """Chispas con estela. kind 0 = absorbidas en espiral hacia una esfera (relativas a su ancla)."""

    def __init__(self, seed=3):
        self.rng = np.random.default_rng(seed)
        self.p = []

    def spawn_inflow(self, anchor_id, R, count):
        for _ in range(count):
            a = self.rng.uniform(0, TAU)
            d = R * self.rng.uniform(2.2, 5.0)
            self.p.append(dict(kind=0, anchor=anchor_id, rel=np.array([np.cos(a), np.sin(a)]) * d,
                               w=self.rng.choice([-1, 1]) * self.rng.uniform(2.0, 5.0),
                               age=0.0, life=self.rng.uniform(0.35, 0.8), b=self.rng.uniform(0.25, 0.8)))

    def spawn_free(self, pos, vel, life, bright, drag=0.0, grav=0.0, behind=False):
        self.p.append(dict(kind=1, pos=np.array(pos, np.float64), vel=np.array(vel, np.float64), age=0.0,
                           life=life, b=bright, drag=drag, grav=grav, behind=behind))

    def step(self, A, B, anchors, dt, pal, apal):
        """Dibuja en A (delante) o en B (detrás de la persona, se enmascara después)."""
        alive = []
        for q in self.p:
            q["age"] += dt
            if q["age"] > q["life"]:
                continue
            fade = smoothstep(0, 0.15, q["age"] / q["life"]) * (1 - smoothstep(0.7, 1.0, q["age"] / q["life"]))
            if q["kind"] == 0:
                anc = anchors.get(q["anchor"])
                if anc is None:
                    continue
                c, R = anc
                prev = c + q["rel"]
                ang = q["w"] * dt
                rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
                dist = np.hypot(*q["rel"])
                q["rel"] = rot @ q["rel"] * max(0.0, 1 - dt * (2.0 + 6.0 * R / max(dist, 1e-3)))
                if np.hypot(*q["rel"]) < 0.5 * R:
                    continue
                cur = c + q["rel"]
                prev = cur + (prev - cur) * 0.5  # estela corta
            else:
                prev = q["pos"].copy()
                q["vel"] = q["vel"] * (1 - q["drag"] * dt) + np.array([0, q["grav"]]) * dt
                q["pos"] = q["pos"] + q["vel"] * dt
                cur = q["pos"]
            behind = q.get("behind", False)
            col = ((apal if behind else pal).fil * 0.7 + pal.core * 0.3) * (q["b"] * fade)
            cv2.line(B if behind else A, tuple(np.round(prev).astype(int)), tuple(np.round(cur).astype(int)),
                     tuple(float(x) for x in col), 1, cv2.LINE_8)
            alive.append(q)
        self.p = alive


def beam(E, heat, O, D, L, w, t, energy, pal, noise):
    """Haz de energía desde O en dirección D: núcleo blanco, envoltura turbulenta que fluye hacia la punta."""
    if energy <= 0.003 or L <= 1:
        return
    h, wd = E.shape[:2]
    X, Y = np.meshgrid(np.arange(wd, dtype=np.float32) - O[0], np.arange(h, dtype=np.float32) - O[1])
    u = X * D[0] + Y * D[1]
    v = -X * D[1] + Y * D[0]
    s = noise.size
    wu = w * (0.55 + 0.45 * smoothstep(0, 0.3 * L + 1, u)) * (1 + 0.04 * np.sin(u / w * 2.0 - t * 31))
    n1 = noise.sample(u / w * 22 - t * 0.9 * s, v / w * 22, 0)
    n2 = noise.sample(u / w * 55 - t * 2.2 * s + 101, v / w * 40, 1)
    vn = np.abs(v) / wu * np.abs(1 + 0.18 * n1) + 0.07 * n2
    ins = smoothstep(-0.3 * w, 0.05 * w, u) * (1 - smoothstep(L - 0.25 * w, L + 0.15 * w, u))
    core = np.exp(-(vn / 0.28) ** 2) * 1.7
    body = np.exp(-(vn / 0.8) ** 4) * 0.55 * (0.85 + 0.15 * np.sin(u / w * 5.5 - t * 45 + n1))
    fil = np.clip(1 - np.abs(n2) * 1.2, 0, 1) ** 6 * np.exp(-(vn / 1.0) ** 2) * 0.7
    ds = np.sqrt(np.maximum(-u, 0) ** 2 + np.maximum(u - L, 0) ** 2 + v * v) / w
    glow = 0.3 / (1 + (ds * 1.5) ** 2)
    E += energy * ((core * ins)[..., None] * pal.core + (body * ins)[..., None] * pal.mid
                   + (fil * ins)[..., None] * pal.fil + glow[..., None] * pal.outer)
    heat += energy * np.exp(-(ds / 1.8) ** 2)


def blast(E, heat, c, R, t, energy, pal, noise):
    """Ataque lanzado hacia la cámara: esfera que se abalanza y llena el cuadro con rayos radiales."""
    if energy <= 0.003:
        return
    orb(E, heat, c, R, t, energy, pal, noise, seed=5, hot=1.3)
    h, w = E.shape[:2]
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32) - c[0], np.arange(h, dtype=np.float32) - c[1])
    r = np.sqrt(X * X + Y * Y) / R + 1e-4
    s = noise.size
    a = (np.arctan2(Y, X) / TAU + 0.5) * s * 4
    rays = np.clip(1 - np.abs(noise.sample(a, np.log(r + 0.2) * 40 - t * 300, 1)) * 0.9, 0, 1) ** 6
    E += energy * (rays * np.exp(-np.maximum(r - 0.8, 0) * 0.9) * 1.4)[..., None] * pal.mid


def ring(E, heat, c, Rr, width, energy, pal):
    """Onda expansiva."""
    if energy <= 0.003:
        return
    r_ = _roi(E.shape, c, Rr + 3 * width)
    if r_ is None:
        return
    sl, X, Y = r_
    d = (np.sqrt(X * X + Y * Y) - Rr) / width
    g = np.exp(-d * d)
    E[sl] += energy * g[..., None] * (pal.mid * 0.6 + pal.core * 0.2)
    heat[sl] += energy * g * 2.0


def aura(E, mask, t, level, pal, noise, dt_hint=1 / 30):
    """Aura tipo llama alrededor de la silueta: plumas que suben, vetas turbulentas, halo y luz de borde."""
    if level <= 0.01:
        return
    h, w = E.shape[:2]
    m = cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    hq, wq = m.shape
    # plumas: arrastre exponencial hacia arriba (duplicando el salto en cada pasada)
    p = m.copy()
    step = max(1, int(hq * 0.0035))
    for i in range(6):
        sh = step * (2 ** i)
        if sh >= hq:
            break
        up = np.zeros_like(p)
        up[:-sh] = p[sh:]
        p = np.maximum(p, up * (0.93 ** (2 ** i)))
    s = noise.size
    xx, yy = np.meshgrid(np.arange(wq, dtype=np.float32), np.arange(hq, dtype=np.float32))
    f = s / (wq * 0.7)
    n1 = noise.sample(xx * f, yy * f + t * s * 0.9, 0)
    n2 = noise.sample(xx * f + 77, yy * f + t * s * 1.2, 1)
    amp = hq * 0.03
    flames = cv2.remap(p, xx + n1 * amp, yy + n2 * amp * 0.7 + amp * 0.5, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    streak = np.clip(1 - np.abs(noise.sample(xx * f * 2.2, yy * f * 0.3 + t * s * 2.0, 0)) * 0.7, 0, 1) ** 2
    flames = cv2.GaussianBlur(flames, (0, 0), 1.2)
    halo = cv2.GaussianBlur(m, (0, 0), hq * 0.035)
    outside = 1 - np.clip(m * 1.3, 0, 1)
    A = (flames * (0.3 + 0.9 * streak) * 0.7 + halo * 0.6) * outside
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(hq * 0.004)))
    rim = cv2.GaussianBlur(np.clip(m - er, 0, 1), (0, 0), 1.5)
    flick = 0.88 + 0.12 * noise.sample(np.float32([[t * 40.0]]), np.float32([[3.0]]), 0)[0, 0]
    A = cv2.resize(A, (w, h), interpolation=cv2.INTER_LINEAR) * flick
    rim = cv2.resize(rim, (w, h), interpolation=cv2.INTER_LINEAR)
    k = np.clip(A, 0, 1)[..., None]
    col = pal.outer * (1 - k) + pal.mid * k
    E += level * (A[..., None] * col * 1.1 + rim[..., None] * (pal.mid * 0.8 + pal.core * 0.3) * 0.9)


def aura_sparks(parts, mask, level, rng, pal, H):
    """Partículas de ki que se desprenden de la silueta y suben."""
    if level <= 0.05:
        return
    m = mask[::4, ::4] > 0.5
    edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(edge)
    if len(xs) == 0:
        return
    for _ in range(rng.poisson(4 * level)):
        i = rng.integers(len(xs))
        x, y = xs[i] * 4.0, ys[i] * 4.0
        parts.spawn_free((x + rng.normal() * 4, y), (rng.normal() * 15, -rng.uniform(0.18, 0.45) * H),
                         rng.uniform(0.3, 0.8), rng.uniform(0.4, 1.0) * level, drag=0.5, behind=True)


def bloom(E, weights=(0.32, 0.27, 0.22, 0.18, 0.15, 0.12)):
    """Dispersión de luz por lente: pirámide gaussiana recombinada. Devuelve (bloom, luz_ambiente)."""
    levels, cur = [], E
    for _ in range(len(weights)):
        if min(cur.shape[:2]) < 8:
            break
        cur = cv2.pyrDown(cur)
        levels.append(cur)
    acc = levels[-1] * weights[len(levels) - 1]
    spill = None
    for i in range(len(levels) - 2, -1, -1):
        acc = cv2.pyrUp(acc, dstsize=(levels[i].shape[1], levels[i].shape[0])) + levels[i] * weights[i]
        if i == 2:
            spill = acc.copy()
    acc = cv2.pyrUp(acc, dstsize=(E.shape[1], E.shape[0]))
    spill = cv2.resize(spill if spill is not None else acc, (E.shape[1], E.shape[0]), interpolation=cv2.INTER_LINEAR)
    return acc, spill


def chroma_shift(E, k=0.004):
    """Aberración cromática leve de lente sobre la luz emitida."""
    h, w = E.shape[:2]
    out = E.copy()
    for ch, s in ((2, 1 + k), (0, 1 - k)):
        M = np.float32([[s, 0, (1 - s) * w / 2], [0, s, (1 - s) * h / 2]])
        out[..., ch] = cv2.warpAffine(E[..., ch], M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out


DECODE = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)


def tonemap_to_u8(X):
    """Luz lineal HDR → 8 bits: lo que pasa de 1 se blanquea y se comprime suavemente (sin recortes duros)."""
    over = np.maximum(X.max(axis=2, keepdims=True) - 1.0, 0)
    X = X + over * 0.45
    k = 0.72
    X = np.where(X < k, X, k + (1 - k) * (1 - np.exp(-(X - k) / (1 - k))))
    return np.clip(cv2.pow(np.maximum(X, 0), 1 / 2.2) * 255 + 0.5, 0, 255).astype(np.uint8)
