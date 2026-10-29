"""Uso:  python -m goku_fx video.mp4 [-o salida.mp4] [--aura dorado] [--timeline tiempos.json] [--debug]"""

import argparse
import os

import numpy as np

from . import timeline
from .fx import PALETTES
from .render import render
from .track import smooth_tracks, track_video


def main():
    ap = argparse.ArgumentParser(description="Bola de energía + Kamehameha + aura siguiendo tus manos.")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", help="video de salida (por defecto <video>_kamehameha.mp4)")
    ap.add_argument("--energia", default="kamehameha", choices=PALETTES, help="color de la energía")
    ap.add_argument("--aura", default="azul", choices=list(PALETTES) + ["no"], help="color del aura, o 'no'")
    ap.add_argument("--timeline", help="JSON con tiempos para corregir la detección automática")
    ap.add_argument("--escala", type=float, default=1.0, help="0.5 = vista previa rápida a media resolución")
    ap.add_argument("--crf", type=int, default=16, help="calidad x264 (menor = mejor)")
    ap.add_argument("--cuadros", type=int, help="renderizar solo los primeros N cuadros")
    ap.add_argument("--debug", action="store_true", help="dibuja el esqueleto de las manos y la fase")
    ap.add_argument("--retrack", action="store_true", help="ignora el análisis guardado y vuelve a analizar")
    ap.add_argument("--sin-audio", action="store_true")
    a = ap.parse_args()

    base = os.path.splitext(a.video)[0]
    out = a.out or base + "_kamehameha.mp4"
    cache = base + ".tracks.npz"
    if os.path.exists(cache) and not a.retrack:
        print(f"Usando análisis guardado: {cache}")
        data = dict(np.load(cache))
    else:
        print("1/3 Siguiendo manos y silueta...")
        data = track_video(a.video)
        np.savez_compressed(cache, **data)

    fps = float(data["fps"])
    hands = data["present"].any(0).mean()
    print(f"    manos detectadas en el {hands:.0%} de los cuadros")
    overrides = timeline.load(a.timeline) if a.timeline else {}
    if a.aura == "no":
        overrides["aura"] = False
    lm, alpha = smooth_tracks(data["lm"], data["present"], fps)
    ev = timeline.detect(lm, alpha, fps, overrides)
    ev_path = base + ".timeline.json"
    timeline.save(ev, ev_path)
    print("2/3 Fases:", timeline.describe(ev))
    print(f"    (puedes editar {ev_path} y pasarlo con --timeline para ajustar)")

    print("3/3 Renderizando efectos...")
    render(a.video, out, data, ev, energy=a.energia, aura_color=a.aura if a.aura != "no" else "azul",
           scale=a.escala, crf=a.crf, debug=a.debug, frames=a.cuadros, keep_audio=not a.sin_audio)


if __name__ == "__main__":
    main()
