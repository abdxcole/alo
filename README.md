# alo · Kamehameha con tus manos

Toma un video tuyo, sigue tus manos cuadro a cuadro y le añade energía estilo Goku:

1. **Carga**: una esfera de plasma en cada mano (núcleo blanco sobreexpuesto, cuerpo turbulento,
   filamentos, descargas eléctricas y chispas que se absorben en espiral).
2. **Fusión**: al juntar las manos las dos energías se conectan con arcos, se funden en una sola
   esfera más grande, con destello y onda expansiva.
3. **Lanzamiento**: al empujar, sale el haz hacia donde mueves las manos (o hacia la cámara si
   empujas al frente), con sacudida de cámara, destello y chispas.
4. **Aura** alrededor de tu silueta: llamas que suben, halo, luz de borde y partículas de ki.

Para que parezca real: todo se compone en luz lineal HDR, la energía **ilumina tu piel y la ropa**
(luz que rebota), hay *bloom* de lente, distorsión por calor, aberración cromática, grano que
coincide con el del video y un tonemap que quema los núcleos a blanco como una cámara.

## Uso

```bash
sudo apt-get install -y libegl1      # MediaPipe lo necesita en Linux
pip install -r requirements.txt

python -m goku_fx mi_video.mp4                       # → mi_video_kamehameha.mp4
python -m goku_fx mi_video.mp4 --aura dorado         # aura Super Saiyajin
python -m goku_fx mi_video.mp4 --escala 0.5          # vista previa rápida
python -m goku_fx mi_video.mp4 --debug               # dibuja el esqueleto de las manos
```

Colores (`--energia` y `--aura`): `kamehameha`, `azul`, `dorado`, `blanco`, `morado`, `rojo`;
`--aura no` lo quita. El audio original se conserva.

## Ajustar los tiempos

Las fases se detectan solas (manos juntas = fusión; empujón brusco = lanzamiento) y se guardan en
`mi_video.timeline.json`. Si algo no cae donde quieres, edítalo y vuelve a renderizar:

```json
{
  "charge_start": 0.4,
  "fuse": 2.1,
  "launch": 3.3,
  "launch_dir": [1.0, -0.1],
  "mode": "beam",
  "beam_duration": 1.6
}
```

```bash
python -m goku_fx mi_video.mp4 --timeline mi_video.timeline.json
```

`mode` puede ser `beam` (haz en la dirección `launch_dir`, en píxeles x/y de pantalla) o `camera`
(el ataque viene hacia el espectador). El análisis de manos se guarda en `mi_video.tracks.npz`, así que
re-renderizar con otros colores o tiempos es rápido.

## Cómo funciona

- `goku_fx/track.py` — MediaPipe Hand Landmarker (21 puntos por mano) y segmentación de persona;
  mantiene la identidad de cada mano, rellena huecos cortos y suaviza sin retraso (es offline).
- `goku_fx/timeline.py` — detecta carga, fusión y lanzamiento a partir de distancia y velocidad de las manos.
- `goku_fx/fx.py` — plasma, rayos, partículas, haz, aura, bloom y tonemap.
- `goku_fx/render.py` — compone cada cuadro y lo codifica con ffmpeg (H.264, audio original).
