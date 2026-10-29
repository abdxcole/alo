"""Composición final cuadro a cuadro: esferas en cada mano → fusión → lanzamiento, más aura y luz real."""

import subprocess

import cv2
import numpy as np

from . import fx
from .fx import smoothstep
from .track import hand_geometry, open_video, smooth_tracks

FX = 2  # los efectos se calculan a la mitad de resolución (son luz difusa) y se escalan


class Plan:
    """Calcula el estado de todos los efectos para un instante, a partir de las manos y la línea de tiempo."""

    def __init__(self, data, ev, sc, fps, fsize):
        lm, self.alpha = smooth_tracks(data["lm"], data["present"], fps)
        c, s = hand_geometry(lm)
        self.center, self.scale = c * sc / FX, s * sc / FX
        self.lm_full = lm[..., :2] * sc
        self.ev, self.fps, (self.fw, self.fh) = ev, fps, fsize
        self.last_mid = None

    def at(self, i):
        ev, s = self.ev, i / self.fps
        al = self.alpha[:, i]
        c, sc = self.center[:, i], np.maximum(self.scale[:, i], 2.0)
        cs, fu, la, dur = ev["charge_start"], ev["fuse"], ev["launch"], ev["beam_duration"]
        st = dict(orbs=[], bridge=None, beam=None, blast=None, rings=[], flash=0.0, burst=0.0)

        charge = float(smoothstep(cs, cs + 1.4, s))
        wsum = al.sum()
        mid = (c * al[:, None]).sum(0) / wsum if wsum > 1e-3 else self.last_mid
        if mid is None:
            mid = np.array([self.fw / 2, self.fh / 2])
        self.last_mid = mid
        both = al.min() > 0.3
        d_norm = np.linalg.norm(c[0] - c[1]) / sc.mean()
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
        R = sc * 0.62 * (0.3 + 0.7 * charge) * np.array([pulse(0), pulse(1)])
        for k in range(2):
            e = al[k] * charge * (1 - mf) ** 0.6 * end_fade
            pos = c[k] * (1 - 0.5 * mf) + mid * 0.5 * mf
            st["orbs"].append(dict(id=k, c=pos, R=R[k] * (1 - 0.45 * mf), e=e, hot=1.0))
        grow = float(smoothstep(fu, fu + 1.5, s)) if fu is not None else 0.0
        Rf = np.sqrt((R ** 2).sum()) * 1.1 * (1 + 0.45 * grow) * pulse(2)
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
            D = np.asarray(ev["launch_dir"], np.float64)
            D = D / (np.linalg.norm(D) + 1e-9)
            if ev["mode"] == "camera":
                Rb = Rf * 1.2 + max(self.fw, self.fh) * 0.75 * (1 - np.exp(-tau / 0.22))
                st["blast"] = dict(c=mid, R=Rb, e=e_b)
            else:
                Lmax = _ray_exit(mid, D, self.fw, self.fh) + Rf
                w = Rf * 0.95 * (0.35 + 0.65 * smoothstep(0, 0.18, tau)) * (0.4 + 0.6 * e_b)
                L = Lmax * (1 - np.exp(-tau / 0.09))
                st["beam"] = dict(O=mid, D=D, L=L, w=w, e=e_b, tip=L < Lmax * 0.97 - Rf)
            st["flash"] += 0.22 * np.exp(-tau / 0.1) + 0.04 * e_b
            if tau < 1.0:
                st["rings"].append(dict(c=mid, R=Rf + tau * self.fh * 2.2, w=self.fh * 0.05 * (1 + tau * 3),
                                        e=0.25 * np.exp(-tau / 0.18)))

        aura_end = 1.0 if tau is None else float(1 - smoothstep(dur, dur + 1.2, tau))
        aura = (0.5 * smoothstep(cs, cs + 0.6, s) + 0.3 * charge + 0.3 * mf_t + 0.4 * e_b) * aura_end
        st["aura"] = float(min(aura, 1.3) * max(al.max(), float(e_b > 0))) if ev.get("aura", True) else 0.0
        st["darken"] = 0.10 * charge * end_fade + 0.08 * mf_t * end_fade + 0.12 * e_b
        st["tremor"] = 0.0008 * charge + 0.0015 * mf_t * end_fade
        st["kick"] = 0.0 if tau is None else 0.012 * np.exp(-tau / 0.35) + 0.0025 * e_b
        st["phase"] = ("lanzamiento" if tau is not None and e_b > 0 else "fusión" if mf_t > 0.5
                       else "carga" if charge > 0 else "—")
        return st


TAU = 2 * np.pi


def _ray_exit(O, D, w, h):
    ts = []
    for o, d, hi in ((O[0], D[0], w), (O[1], D[1], h)):
        if abs(d) > 1e-6:
            ts.append(((hi if d > 0 else 0) - o) / d)
    return max(0.0, min(ts)) if ts else max(w, h)


def _writer(path, size, fps, audio_src, crf):
    w, h = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps:.6f}", "-i", "-"]
    if audio_src:
        cmd += ["-i", audio_src, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]


def render(video, out_path, data, ev, energy="kamehameha", aura_color="azul", scale=1.0, crf=16,
           debug=False, frames=None, keep_audio=True):
    cap, fps = open_video(video)
    W, H = (int(round(v * scale)) // 2 * 2 for v in data["size"])
    fw, fh = W // FX, H // FX
    plan = Plan(data, ev, scale, fps, (fw, fh))
    pal, apal = fx.Palette(energy), fx.Palette(aura_color)
    noise = fx.Noise()
    parts = fx.Particles()
    rng = np.random.default_rng(1)
    grain = [np.random.default_rng(k).normal(size=(H, W)).astype(np.float32) for k in range(3)]
    xx, yy = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    masks = data["masks"]
    n = min(len(masks), plan.alpha.shape[1])
    if frames:
        n = min(n, frames)
    proc = _writer(out_path, (W, H), fps, video if keep_audio else None, crf)
    mask_s = None
    dt = 1.0 / fps

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        s = i / fps
        st = plan.at(i)
        E = np.zeros((fh, fw, 3), np.float32)
        heat = np.zeros((fh, fw), np.float32)
        A = np.zeros_like(E)
        B = np.zeros_like(E)

        m = cv2.resize(masks[i].astype(np.float32) / 255.0, (fw, fh), interpolation=cv2.INTER_LINEAR)
        mask_s = m if mask_s is None else mask_s * 0.35 + m * 0.65
        fx.aura(E, mask_s, s, st["aura"], apal, noise)
        fx.aura_sparks(parts, mask_s, st["aura"], rng, apal, fh)

        anchors = {}
        for o in st["orbs"]:
            if o["e"] > 0.003:
                fx.orb(E, heat, o["c"], o["R"], s, o["e"], pal, noise, seed=o["id"], hot=o["hot"])
                fx.orb_bolts(A, o["c"], o["R"], o["e"], rng, pal, rate=2.0 + 10 * st["burst"])
                parts.spawn_inflow(o["id"], o["R"], rng.poisson(2.0 * o["e"] * (1 + (o["id"] == 2))))
                anchors[o["id"]] = (o["c"], o["R"])
        if st["bridge"] is not None and st["bridge"][2] > 0.02:
            pa, pb, k = st["bridge"]
            for _ in range(rng.poisson(3 * k)):
                fx.draw_bolt(A, pa, pb, rng, (pal.fil * 0.7 + pal.core * 0.3) * rng.uniform(0.8, 1.6) * k, rough=0.22)
            d = np.linalg.norm(pb - pa)
            if d > 1:
                fx.beam(E, heat, pa, (pb - pa) / d, d, min(o["R"] for o in st["orbs"][:2]) * 0.35 + 1, s,
                        k * 0.6, pal, noise)
        b = st["beam"]
        if b is not None:
            fx.beam(E, heat, b["O"], b["D"], b["L"], b["w"], s, b["e"], pal, noise)
            fx.orb(E, heat, b["O"], b["w"] * 1.35, s, b["e"], pal, noise, seed=7, hot=1.5)
            if b["tip"]:
                fx.orb(E, heat, b["O"] + b["D"] * b["L"], b["w"] * 1.3, s * 1.7, b["e"], pal, noise, seed=9, hot=1.2)
            nrm = np.array([-b["D"][1], b["D"][0]])
            for _ in range(rng.poisson(7 * b["e"])):
                u = rng.uniform(0, b["L"])
                side = rng.choice([-1, 1])
                p = b["O"] + b["D"] * u + nrm * side * b["w"] * rng.uniform(0.6, 1.1)
                v = b["D"] * rng.uniform(0.3, 0.9) * fh + nrm * side * rng.uniform(0.05, 0.3) * fh
                parts.spawn_free(p, v, rng.uniform(0.15, 0.4), rng.uniform(0.6, 1.4) * b["e"], drag=2.0)
            for _ in range(rng.poisson(2.5 * b["e"])):
                u0 = rng.uniform(0, b["L"] * 0.9)
                p0 = b["O"] + b["D"] * u0 + nrm * rng.normal() * b["w"] * 0.3
                p1 = p0 + b["D"] * b["w"] * rng.uniform(0.5, 2) + nrm * rng.choice([-1, 1]) * b["w"] * rng.uniform(1.0, 2.0)
                fx.draw_bolt(A, p0, p1, rng, pal.fil * rng.uniform(0.7, 1.4) * b["e"])
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

        bl_, spill = fx.bloom(E)
        glow = fx.chroma_shift(E + bl_)
        F = fx.DECODE[frame]
        F *= 1 - st["darken"]
        F *= 1 + cv2.resize(spill, (W, H), interpolation=cv2.INTER_LINEAR) * 1.4  # luz que rebota en piel y ropa
        G = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
        lum = np.clip(G.max(axis=2), 0, 1)
        F += G + (grain[i % 3] * lum * 0.012)[..., None]
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
            _draw_debug(out, plan, i, st, s)
        proc.stdin.write(out.tobytes())
        if i % 15 == 0:
            print(f"  render: {i}/{n} [{st['phase']}]", end="\r", flush=True)

    proc.stdin.close()
    proc.wait()
    cap.release()
    print(f"  render: {n}/{n} listo → {out_path}")


def _draw_debug(img, plan, i, st, s):
    for k in range(2):
        if plan.alpha[k, i] < 0.3:
            continue
        pts = plan.lm_full[k, i].astype(int)
        col = (0, 255, 0) if k == 0 else (0, 200, 255)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img, f"{s:5.2f}s  {st['phase']}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3,
                cv2.LINE_AA)
