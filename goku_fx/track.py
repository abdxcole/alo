"""Pasada de análisis: manos (21 puntos por mano) y silueta de la persona, cuadro a cuadro."""

import os
import urllib.request

import cv2
import numpy as np

MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task",
    "selfie_multiclass.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}
MASK_SCALE = 4  # la silueta se guarda a 1/4 de resolución
MAX_GAP_S = 0.5  # huecos de detección más cortos que esto se interpolan


def model_path(name):
    cache = os.path.join(os.path.expanduser("~"), ".cache", "goku_fx")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        print(f"Descargando modelo {name}...")
        urllib.request.urlretrieve(MODELS[name], path + ".part")
        os.replace(path + ".part", path)
    return path


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, fps


def track_video(path, max_hands=2):
    """Devuelve dict con landmarks crudos por pista (A/B), presencia, máscaras y metadatos."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
        min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35,
        min_tracking_confidence=0.35))
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path("selfie_multiclass.tflite")),
        running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True))

    cap, fps = open_video(path)
    raw, masks = [], []
    i = 0
    h = w = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(round(i * 1000.0 / fps))

        res = hands.detect_for_video(img, ts)
        dets = []
        for lms, hd in zip(res.hand_landmarks, res.handedness):
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], np.float32)
            dets.append((pts, hd[0].score))
        raw.append(dets)

        sm = seg.segment_for_video(img, ts)
        bg = sm.confidence_masks[0].numpy_view()  # categoría 0 = fondo
        person = 1.0 - np.squeeze(bg)
        small = cv2.resize(person, (w // MASK_SCALE, h // MASK_SCALE), interpolation=cv2.INTER_AREA)
        masks.append(np.clip(small * 255, 0, 255).astype(np.uint8))

        i += 1
        if i % 30 == 0:
            print(f"  análisis: {i} cuadros", end="\r", flush=True)
    cap.release()
    hands.close()
    seg.close()
    print(f"  análisis: {i} cuadros listos")

    lm, present = assign_tracks(raw, i)
    return {"lm": lm, "present": present, "masks": np.stack(masks), "fps": np.float32(fps),
            "size": np.array([w, h], np.int32)}


def assign_tracks(raw, n):
    """Mantiene la identidad de cada mano entre cuadros (pista A = la que empieza más a la izquierda)."""
    lm = np.zeros((2, n, 21, 3), np.float32)
    present = np.zeros((2, n), bool)
    last = [None, None]
    for t, dets in enumerate(raw):
        dets = [d for d in dets if d[1] > 0.5 and _hand_scale(d[0]) > 12]
        if not dets:
            continue
        centers = [d[0][:, :2].mean(0) for d in dets]
        if last[0] is None and last[1] is None:
            order = np.argsort([c[0] for c in centers])
            slots = [0, 1] if len(dets) == 2 else [0]
            pairs = list(zip(slots, order))
        elif len(dets) == 2:
            cost = lambda a, b: _dist(last[a], centers[b])
            if cost(0, 0) + cost(1, 1) <= cost(0, 1) + cost(1, 0):
                pairs = [(0, 0), (1, 1)]
            else:
                pairs = [(0, 1), (1, 0)]
        else:
            k = int(np.argmin([_dist(last[0], centers[0]), _dist(last[1], centers[0])]))
            pairs = [(k, 0)]
        for slot, d in pairs:
            lm[slot, t] = dets[d][0]
            present[slot, t] = True
            last[slot] = centers[d]
    return lm, present


def _dist(a, b):
    return 1e9 if a is None else float(np.linalg.norm(a - b))


def _hand_scale(pts):
    return float(np.linalg.norm(pts[0, :2] - pts[9, :2]))


def smooth_tracks(lm, present, fps, sigma_s=0.045):
    """Interpola huecos cortos y suaviza sin retraso (filtro gaussiano centrado, es un proceso offline).

    Devuelve landmarks suavizados y un alfa de presencia 0..1 con fundidos suaves."""
    n = lm.shape[1]
    out = lm.copy()
    alpha = np.zeros((2, n), np.float32)
    max_gap = int(MAX_GAP_S * fps)
    for k in range(2):
        idx = np.flatnonzero(present[k])
        if len(idx) == 0:
            continue
        valid = present[k].copy()
        # rellena huecos internos cortos
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= max_gap + 1:
                f = np.linspace(0, 1, b - a + 1)[1:-1, None, None]
                out[k, a + 1:b] = out[k, a] * (1 - f) + out[k, b] * f
                valid[a + 1:b] = True
        # suavizado por segmentos contiguos
        sig = max(sigma_s * fps, 0.6)
        for s, e in _runs(valid):
            seg = out[k, s:e].reshape(e - s, -1)
            out[k, s:e] = _gauss1d(seg, sig).reshape(e - s, 21, 3)
        alpha[k] = _gauss1d(valid.astype(np.float32)[:, None], 0.08 * fps)[:, 0]
        alpha[k] = np.where(valid, np.maximum(alpha[k], 0.5), alpha[k])
        alpha[k] = np.clip((alpha[k] - 0.15) / 0.7, 0, 1)
        # mantiene la última posición conocida mientras se desvanece
        last = None
        for t in range(n):
            if valid[t]:
                last = out[k, t]
            elif last is not None:
                out[k, t] = last
        first = idx[0]
        out[k, :first] = out[k, first]
    return out, alpha


def _runs(mask):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _gauss1d(x, sigma):
    """Filtro gaussiano a lo largo del eje 0 con bordes replicados."""
    if sigma <= 0 or len(x) < 2:
        return x
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(x[:1], r, 0), x, np.repeat(x[-1:], r, 0)])
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(pad[:, c], k, mode="valid")
    return out


def hand_geometry(lm):
    """Centro donde se forma la energía (entre palma y dedos) y escala de la mano en px."""
    palm = lm[..., [0, 5, 9, 13, 17], :2].mean(-2)
    tips = lm[..., [4, 8, 12, 16, 20], :2].mean(-2)
    center = palm * 0.55 + tips * 0.45
    scale = np.linalg.norm(lm[..., 0, :2] - lm[..., 9, :2], axis=-1)
    return center, scale
