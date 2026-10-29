"""Preparación del video: HDR de iPhone → SDR, y cuadros intermedios para la cámara lenta."""

import json
import os
import subprocess

HDR_TRANSFERS = {"arib-std-b67", "smpte2084"}


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=color_transfer,color_primaries", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    streams = json.loads(out).get("streams") or [{}]
    return streams[0]


def ensure_sdr(path, base):
    """Los videos HDR (HLG/PQ) se ven lavados si se procesan tal cual: se convierten a Rec.709 con tonemap."""
    info = probe(path)
    trc = info.get("color_transfer", "")
    if trc not in HDR_TRANSFERS:
        return path
    out = base + ".sdr.mp4"
    if not os.path.exists(out):
        print("    video HDR detectado: convirtiendo a SDR...")
        src = f"tin={trc}:min=bt2020nc:pin={info.get('color_primaries', 'bt2020')}:rin=tv"
        vf = (f"zscale={src}:t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,"
              "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-map", "0:v:0", "-map", "0:a:0?",
                        "-vf", vf, "-c:v", "libx264", "-crf", "12", "-preset", "medium", "-c:a", "copy",
                        out + ".tmp.mp4"], check=True)
        os.replace(out + ".tmp.mp4", out)
    return out


def interpolated(path, base, fps=120):
    """Cuadros intermedios por compensación de movimiento para que la cámara lenta sea fluida."""
    out = base + f".{fps}fps.mp4"
    if not os.path.exists(out):
        print(f"    generando cuadros intermedios a {fps} fps (tarda unos minutos)...")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-an", "-vf",
                        f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                        "-c:v", "libx264", "-crf", "12", "-preset", "fast", out + ".tmp.mp4"], check=True)
        os.replace(out + ".tmp.mp4", out)
    return out
