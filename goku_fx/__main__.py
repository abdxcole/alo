"""Uso:  python -m goku_fx video.mp4 [-o salida.mp4] [--aura dorado] [--lento rampa] [--timeline tiempos.json]"""

import argparse
import os
import subprocess

import numpy as np

from . import prep, retime, timeline
from .fx import PALETTES
from .render import render
from .track import smooth_tracks, track_video


def main():
    ap = argparse.ArgumentParser(description="Bola de energía + Kamehameha + aura siguiendo tus manos.")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", help="video de salida (por defecto <video>_kamehameha.mp4)")
    ap.add_argument("--energia", default="kamehameha", choices=PALETTES, help="color de la energía")
    ap.add_argument("--aura", default="azul", choices=list(PALETTES) + ["no"], help="color del aura, o 'no'")
    ap.add_argument("--lento", default="rampa",
                    help="cámara lenta: 'rampa' (frena en la fusión y el ataque), 'no', o un factor fijo como 0.5")
    ap.add_argument("--timeline", help="JSON con tiempos para corregir la detección automática")
    ap.add_argument("--escala", type=float, default=1.0, help="0.5 = vista previa rápida a media resolución")
    ap.add_argument("--crf", type=int, default=16, help="calidad x264 (menor = mejor)")
    ap.add_argument("--debug", action="store_true", help="dibuja el esqueleto de las manos y la fase")
    ap.add_argument("--retrack", action="store_true", help="ignora el análisis guardado y vuelve a analizar")
    ap.add_argument("--sin-sonido", action="store_true", help="no añadir efectos de sonido")
    a = ap.parse_args()

    base = os.path.splitext(a.video)[0]
    out = a.out or base + "_kamehameha.mp4"
    video = prep.ensure_sdr(a.video, base)
    cache = base + ".tracks.npz"
    if os.path.exists(cache) and not a.retrack:
        print(f"Usando análisis guardado: {cache}")
        data = dict(np.load(cache))
    else:
        print("1/4 Siguiendo manos y silueta...")
        data = track_video(video)
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
    print("2/4 Fases:", timeline.describe(ev))
    print(f"    (puedes editar {ev_path} y pasarlo con --timeline para ajustar)")

    duration = len(data["masks"]) / fps
    timemap = retime.build(ev, duration, fps, a.lento)
    source = video if a.lento == "no" else prep.interpolated(video, base)
    print(f"3/4 Renderizando efectos ({len(timemap)} cuadros, {len(timemap) / fps:.1f} s)...")
    silent = out + ".video.mp4"
    log = render(source, silent, data, ev, timemap, fps, energy=a.energia,
                 aura_color=a.aura if a.aura != "no" else "azul", scale=a.escala, crf=a.crf, debug=a.debug)

    if a.sin_sonido:
        os.replace(silent, out)
    else:
        print("4/4 Sintetizando efectos de sonido...")
        from .sfx import synthesize
        wav = synthesize(log, fps, out + ".sfx.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out],
                       check=True)
        os.remove(silent)
        os.remove(wav)
    print(f"Listo → {out}")


if __name__ == "__main__":
    main()
