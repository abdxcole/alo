"""Composición final cuadro a cuadro: esferas en cada mano → fusión → lanzamiento, más aura y luz real.

El tiempo de la historia (τ, en segundos del video original) se separa del de salida, así la cámara
lenta frena también la energía: chispas, rayos y sacudidas van al mismo ritmo que el cuerpo."""

import subprocess

import cv2
import numpy as np

from . import fx
from .fx import smoothstep
from .track import _gauss1d, _runs, open_video, smooth_tracks

FX = 2  # los efectos se calculan a la mitad de resolución (son luz difusa) y se escalan
TAU = 2 * np.pi
PALM = [0, 5, 9, 13, 17]


def _lerp(arr, fi, axis=0):
    """Interpola linealmente un arreglo en un índice de cuadro fraccionario."""
    n = arr.shape[axis]
    i0 = int(np.clip(np.floor(fi), 0, n - 1))
    i1 = min(i0 + 1, n - 1)
    f = float(np.clip(fi - i0, 0, 1))
    a, b = np.take(arr, i0, axis), np.take(arr, i1, axis)
    return a * (1 - f) + b * f


def _hover(pts, P, d, R, need=1.0):
    """Desplaza el centro desde P en dirección d hasta que la esfera no toque ningún punto de la mano."""
    for t in np.linspace(0, 3.5, 36) * R:
        C = P + t * d
        if np.min(np.linalg.norm(pts - C, axis=1)) >= need * R:
            return C
    return P + 3.5 * R * d


def _float_dir(lm, vis):
    """Hacia dónde flota la energía: entre la dirección de los dedos y 'arriba' en pantalla."""
    f = sum((lm[k, 9] - lm[k, 0]) for k in range(len(lm)) if vis[k])
    f = f / (np.linalg.norm(f) + 1e-6)
    d = 0.55 * f + np.array([0.0, -0.6])
    n = np.linalg.norm(d)
    return d / n if n > 0.2 else f


class Plan:
    """Calcula el estado de todos los efectos para un instante τ a partir de las manos y la línea de tiempo."""

    def __init__(self, data, ev, sc, fps, fsize):
        lm, self.alpha = smooth_tracks(data["lm"], data["present"], fps)
        self.lm_full = lm[..., :2] * sc
        L = self.lm_full / FX
        self.ev, self.fps, (self.fw, self.fh) = ev, fps, fsize
        n = L.shape[1]
        self.n = n
        scale = np.maximum(np.linalg.norm(L[:, :, 0] - L[:, :, 9], axis=-1), 2.0)
        cs = ev["charge_start"]
        charge = smoothstep(cs, cs + 1.4, np.arange(n) / fps)
        # radio base de cada esfera: proporcional a la mano, con un mínimo para manos lejanas
        self.R = np.maximum(scale * 0.6, 0.03 * self.fh) * (0.3 + 0.7 * charge)
        fu = ev["fuse"]
        grow = smoothstep(fu, fu + 1.5, np.arange(n) / fps) if fu is not None else np.zeros(n)
        self.Rf = np.sqrt((self.R ** 2).sum(0)) * 1.1 * (1 + 0.45 * grow)

        # ancla de cada esfera: flota sobre la mano, sin tocarla
        vis = self.alpha > 0.3
        self.anchor = np.zeros((2, n, 2), np.float32)
        self.fused = np.zeros((n, 2), np.float32)
        for t in range(n):
            for k in range(2):
                P = L[k, t, PALM].mean(0)
                self.anchor[k, t] = _hover(L[k, t], P, _float_dir(L[:, t], [k == 0, k == 1]), self.R[k, t])
            both = vis[0, t] and vis[1, t]
            ks = [k for k in range(2) if vis[k, t]] or [int(np.argmax(self.alpha[:, t]))]
            pts = np.concatenate([L[k, t] for k in ks])
            P = np.mean([L[k, t, PALM].mean(0) for k in ks], axis=0)
            self.fused[t] = _hover(pts, P, _float_dir(L[:, t], [0 in ks, 1 in ks]), self.Rf[t], 0.95 if both else 1.0)
        for k in range(2):
            for s0, e0 in _runs(vis[k]):
                self.anchor[k, s0:e0] = _gauss1d(self.anchor[k, s0:e0], 1.5)
        self.fused = _gauss1d(self.fused, 2.5)
        self.palm = L[:, :, PALM].mean(2)
        self.scale = scale
        self.last_mid = None
        self.origin = None

    def at(self, s):
        ev = self.ev
        fi = s * self.fps
        al = _lerp(self.alpha, fi, 1)
        anchor = _lerp(self.anchor, fi, 1)
        palm = _lerp(self.palm, fi, 1)
        sc = _lerp(self.scale, fi, 1)
        Rb = _lerp(self.R, fi, 1)
        Rf = float(_lerp(self.Rf, fi))
        fused = _lerp(self.fused, fi)
        cs, fu, la, dur = ev["charge_start"], ev["fuse"], ev["launch"], ev["beam_duration"]
        st = dict(orbs=[], bridge=None, beam=None, blast=None, rings=[], flash=0.0, burst=0.0)

        charge = float(smoothstep(cs, cs + 1.4, s))
        mid = fused if al.max() > 1e-3 else self.last_mid
        if mid is None:
            mid = np.array([self.fw / 2, self.fh / 2])
        self.last_mid = mid
        both = al.min() > 0.3
        d_norm = np.linalg.norm(palm[0] - palm[1]) / sc.mean()
        mf_d = float(smoothstep(3.4, 1.9, d_norm)) if both else 0.0
        mf_t = float(smoothstep(fu - 0.05, fu + 0.22, s)) if fu is not None else 0.0
        mf = max(0.85 * mf_d, mf_t)

        # fase de lanzamiento
        e_b, tau = 0.0, None
        if la is not None and s >= la:
            tau = s - la
            e_b = float(smoothstep(0, 0.05, tau) * (1 - smoothstep(dur, dur + 0.45, tau)))
        end_fade = 1.0 if tau is None else float(1 - smoothstep(dur, dur + 0.45, tau))

        pulse = lambda k: 1 + 0.05 * np.sin(TAU * 6.3 * s + k * 2.1) + 0.03 * np.sin(TAU * 13.1 * s + k)
        R = Rb * np.array([pulse(0), pulse(1)])
        for k in range(2):
            e = al[k] * charge * (1 - mf) ** 0.6 * end_fade
            pos = anchor[k] * (1 - mf) + mid * mf
            st["orbs"].append(dict(id=k, c=pos, R=R[k] * (1 - 0.45 * mf), e=e, hot=1.0))
        grow = float(smoothstep(fu, fu + 1.5, s)) if fu is not None else 0.0
        Rf = Rf * pulse(2)
        if tau is not None:
            Rf *= 1 + 0.3 * smoothstep(0, 0.1, tau)
        st["orbs"].append(dict(id=2, c=mid, R=Rf, e=mf * charge * max(al.max(), 0.0 if tau is None else 1.0) * end_fade,
                               hot=1 + 0.4 * grow + 0.4 * e_b))
        if both and d_norm < 4.2 and mf < 0.98:
            st["bridge"] = (st["orbs"][0]["c"], st["orbs"][1]["c"],
                            float(smoothstep(4.2, 2.2, d_norm)) * (1 - mf) * charge)

        if fu is not None and s >= fu:
            tf = s - fu
            st["flash"] += 0.15 * np.exp(-tf / 0.12)
            st["burst"] = float(np.exp(-tf / 0.25))
            if tf < 0.8:
                st["rings"].append(dict(c=mid, R=Rf + tf * self.fh * 0.9, w=self.fh * 0.02 * (1 + tf * 3),
                                        e=0.25 * np.exp(-tf / 0.2)))

        if tau is not None:
            # el origen del ataque sigue a las manos con inercia (no se cae cuando bajas los brazos)
            follow = 0.06 if ev["mode"] == "camera" else 0.25
            self.origin = mid.copy() if self.origin is None else self.origin + follow * (mid - self.origin)
            mid = self.origin
            D = np.asarray(ev["launch_dir"], np.float64)
            D = D / (np.linalg.norm(D) + 1e-9)
            if ev["mode"] == "camera":
                Rbl = Rf * 1.2 + max(self.fw, self.fh) * 0.22 * (1 - np.exp(-tau / 0.18))
                st["blast"] = dict(c=mid, R=Rbl, e=e_b)
            else:
                Lmax = _ray_exit(mid, D, self.fw, self.fh) + Rf
                w = Rf * 0.95 * (0.35 + 0.65 * smoothstep(0, 0.18, tau)) * (0.4 + 0.6 * e_b)
                Lb = Lmax * (1 - np.exp(-tau / 0.09))
                st["beam"] = dict(O=mid, D=D, L=Lb, w=w, e=e_b, tip=Lb < Lmax * 0.97 - Rf)
            st["flash"] += 0.22 * np.exp(-tau / 0.1) + 0.04 * e_b
            if tau < 1.0:
                st["rings"].append(dict(c=mid, R=Rf + tau * self.fh * 2.2, w=self.fh * 0.05 * (1 + tau * 3),
                                        e=0.25 * np.exp(-tau / 0.18)))

        aura_end = 1.0 if tau is None else float(1 - smoothstep(dur, dur + 1.2, tau))
        aura = (0.5 * smoothstep(cs, cs + 0.6, s) + 0.3 * charge + 0.3 * mf_t + 0.4 * e_b) * aura_end
        st["aura"] = float(min(aura, 1.3)) if ev.get("aura", True) else 0.0
        st["darken"] = 0.14 * charge * end_fade + 0.08 * mf_t * end_fade + 0.12 * e_b
        st["tremor"] = 0.0008 * charge + 0.0015 * mf_t * end_fade
        st["kick"] = 0.0 if tau is None else 0.012 * np.exp(-tau / 0.35) + 0.0025 * e_b
        st["e_b"] = e_b
        st["phase"] = ("lanzamiento" if tau is not None and e_b > 0 else "fusión" if mf_t > 0.5
                       else "carga" if charge > 0 else "—")
        return st


def _ray_exit(O, D, w, h):
    ts = []
    for o, d, hi in ((O[0], D[0], w), (O[1], D[1], h)):
        if abs(d) > 1e-6:
            ts.append(((hi if d > 0 else 0) - o) / d)
    return max(0.0, min(ts)) if ts else max(w, h)


def _writer(path, size, fps, crf):
    w, h = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


class FrameSource:
    """Entrega el cuadro de la fuente más cercano a un instante τ (lectura secuencial, τ creciente)."""

    def __init__(self, path):
        self.cap, self.fps = open_video(path)
        self.idx, self.frame = -1, None

    def at(self, s):
        want = int(round(s * self.fps))
        while self.idx < want:
            ok, f = self.cap.read()
            if not ok:
                break
            self.idx, self.frame = self.idx + 1, f
        return self.frame


def render(source, out_path, data, ev, timemap, out_fps, energy="kamehameha", aura_color="azul", scale=1.0,
           crf=16, debug=False):
    """source: video del que salen los cuadros (puede ser una versión interpolada a más fps).
    timemap: τ (segundos de la historia) de cada cuadro de salida. Devuelve el registro por cuadro para el sonido."""
    fps = float(data["fps"])
    W, H = (int(round(v * scale)) // 2 * 2 for v in data["size"])
    fw, fh = W // FX, H // FX
    plan = Plan(data, ev, scale, fps, (fw, fh))
    src = FrameSource(source)
    pal, apal = fx.Palette(energy), fx.Palette(aura_color)
    noise = fx.Noise()
    parts = fx.Particles()
    rng = np.random.default_rng(1)
    grain = [np.random.default_rng(k).normal(size=(H, W)).astype(np.float32) for k in range(3)]
    xx, yy = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    masks = data["masks"].astype(np.float32) / 255.0
    proc = _writer(out_path, (W, H), out_fps, crf)
    mask_s = None
    log = []
    n = len(timemap)

    for j, s in enumerate(timemap):
        frame = src.at(s)
        if frame is None:
            break
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        dt = s - timemap[j - 1] if j > 0 else 1.0 / fps
        rate = dt * fps  # cuántos cuadros "de historia" avanzan en este cuadro de salida
        st = plan.at(s)
        # los rayos cambian al ritmo de la historia: en cámara lenta duran más en pantalla
        brng = np.random.default_rng(int(s * fps) * 7919 + 13)
        E = np.zeros((fh, fw, 3), np.float32)
        heat = np.zeros((fh, fw), np.float32)
        A = np.zeros_like(E)
        B = np.zeros_like(E)
        bolts = 0

        m = cv2.resize(_lerp(masks, s * fps), (fw, fh), interpolation=cv2.INTER_LINEAR)
        k = min(1.0, 0.65 * rate)
        mask_s = m if mask_s is None else mask_s * (1 - k) + m * k
        Ea = np.zeros_like(E)
        fx.aura(Ea, mask_s, s, st["aura"], apal, noise)
        fx.aura_sparks(parts, mask_s, st["aura"], rng, apal, fh, rate)

        anchors = {}
        for o in st["orbs"]:
            if o["e"] > 0.003:
                fx.orb(E, heat, o["c"], o["R"], s, o["e"], pal, noise, seed=o["id"], hot=o["hot"])
                bolts += fx.orb_bolts(A, o["c"], o["R"], o["e"], brng, pal, rate=2.0 + 10 * st["burst"])
                parts.spawn_inflow(o["id"], o["R"], rng.poisson(2.0 * o["e"] * (1 + (o["id"] == 2)) * rate))
                anchors[o["id"]] = (o["c"], o["R"])
        if st["bridge"] is not None and st["bridge"][2] > 0.02:
            pa, pb, kb = st["bridge"]
            for _ in range(brng.poisson(3 * kb)):
                fx.draw_bolt(A, pa, pb, brng, (pal.fil * 0.7 + pal.core * 0.3) * brng.uniform(0.8, 1.6) * kb, rough=0.22)
                bolts += 1
            d = np.linalg.norm(pb - pa)
            if d > 1:
                fx.beam(E, heat, pa, (pb - pa) / d, d, min(o["R"] for o in st["orbs"][:2]) * 0.35 + 1, s,
                        kb * 0.6, pal, noise)
        b = st["beam"]
        if b is not None:
            fx.beam(E, heat, b["O"], b["D"], b["L"], b["w"], s, b["e"], pal, noise)
            fx.orb(E, heat, b["O"], b["w"] * 1.35, s, b["e"], pal, noise, seed=7, hot=1.5)
            if b["tip"]:
                fx.orb(E, heat, b["O"] + b["D"] * b["L"], b["w"] * 1.3, s * 1.7, b["e"], pal, noise, seed=9, hot=1.2)
            nrm = np.array([-b["D"][1], b["D"][0]])
            for _ in range(rng.poisson(7 * b["e"] * rate)):
                u = rng.uniform(0, b["L"])
                side = rng.choice([-1, 1])
                p = b["O"] + b["D"] * u + nrm * side * b["w"] * rng.uniform(0.6, 1.1)
                v = b["D"] * rng.uniform(0.3, 0.9) * fh + nrm * side * rng.uniform(0.05, 0.3) * fh
                parts.spawn_free(p, v, rng.uniform(0.15, 0.4), rng.uniform(0.6, 1.4) * b["e"], drag=2.0)
            for _ in range(brng.poisson(2.5 * b["e"])):
                u0 = brng.uniform(0, b["L"] * 0.9)
                p0 = b["O"] + b["D"] * u0 + nrm * brng.normal() * b["w"] * 0.3
                p1 = p0 + b["D"] * b["w"] * brng.uniform(0.5, 2) + nrm * brng.choice([-1, 1]) * b["w"] * brng.uniform(1.0, 2.0)
                fx.draw_bolt(A, p0, p1, brng, pal.fil * brng.uniform(0.7, 1.4) * b["e"])
                bolts += 1
        if st["blast"] is not None:
            bl = st["blast"]
            fx.blast(E, heat, bl["c"], bl["R"], s, bl["e"], pal, noise)
        for r in st["rings"]:
            fx.ring(E, heat, r["c"], r["R"], r["w"], r["e"], pal)

        parts.step(A, B, anchors, dt, pal, apal)
        E += A * 0.9 + cv2.GaussianBlur(A, (0, 0), 1.2) * 0.9
        E += (B * 1.2 + cv2.GaussianBlur(B, (0, 0), 1.2) * 1.2) * (1 - np.clip(mask_s * 1.5, 0, 1))[..., None]

        # distorsión por calor alrededor de la energía
        if heat.max() > 0.02:
            ns = 1.3  # ~1 texel por píxel: ondas de calor suaves, no ruido de arena
            dx = noise.sample(xx * ns, yy * ns + s * 160, 0) * heat
            dy = noise.sample(xx * ns + 51, yy * ns + s * 160, 0) * heat
            amp = H * 0.004
            dx = cv2.resize(dx, (W, H)) * amp
            dy = cv2.resize(dy, (W, H)) * amp
            frame = cv2.remap(frame, gx + dx, gy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        _, spill = fx.bloom(E)
        E += Ea
        bl_, _ = fx.bloom(E)
        glow = fx.chroma_shift(E + bl_)
        F = fx.DECODE[frame]
        F *= 1 - st["darken"]
        F *= 1 + cv2.resize(spill, (W, H), interpolation=cv2.INTER_LINEAR) * 1.2  # luz que rebota en piel y ropa
        G = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
        lum = np.clip(G.max(axis=2), 0, 1)
        F += G + (grain[j % 3] * lum * 0.012)[..., None]
        if st["flash"] > 0.002:
            F += st["flash"] * (pal.mid * 0.5 + pal.core * 0.3)
        out = fx.tonemap_to_u8(F)

        amp = H * (st["tremor"] + st["kick"])
        if amp > 0.3:
            ox = amp * float(noise.sample(np.float32([[s * 900.0]]), np.float32([[5.0]]), 0)[0, 0])
            oy = amp * float(noise.sample(np.float32([[s * 900.0]]), np.float32([[205.0]]), 1)[0, 0])
            M = np.float32([[1, 0, ox], [0, 1, oy]])
            out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        if debug:
            _draw_debug(out, plan, s, st)
        proc.stdin.write(out.tobytes())
        log.append(dict(s=s, rate=rate, aura=st["aura"], orb=sum(o["e"] for o in st["orbs"]),
                        fused=st["orbs"][2]["e"], bolts=bolts, flash=st["flash"], beam=st["e_b"],
                        burst=st["burst"]))
        if j % 15 == 0:
            print(f"  render: {j}/{n} [{st['phase']}]", end="\r", flush=True)

    proc.stdin.close()
    proc.wait()
    print(f"  render: {n}/{n} listo")
    return log


def _draw_debug(img, plan, s, st):
    fi = s * plan.fps
    al = _lerp(plan.alpha, fi, 1)
    lm = _lerp(plan.lm_full, fi, 1)
    for k in range(2):
        if al[k] < 0.3:
            continue
        pts = lm[k].astype(int)
        col = (0, 255, 0) if k == 0 else (0, 200, 255)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img, f"{s:5.2f}s  {st['phase']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                cv2.LINE_AA)
