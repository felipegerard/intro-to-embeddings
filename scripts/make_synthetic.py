"""Genera el dataset sintético: glifos al azar, una carpeta por identidad.

Cada identidad es un glifo hecho de 2 a 4 trazos. Cada imagen de esa identidad
es el mismo glifo dibujado con variaciones: rotación, escala, desplazamiento,
temblor en los trazos y grosor. `--variation` escala todas esas variaciones a
la vez: con 0, todas las imágenes de una identidad son idénticas; con 1, todavía
se reconocen, pero cuesta. `--noise` agrega ruido de píxel por separado.

Con los valores por defecto se genera el dataset común de todos los equipos: la
columna comparable de la sesión 3. No los cambien para esa corrida; para sus
propios experimentos, generen otro en otra carpeta.

    python scripts/make_synthetic.py
    python scripts/make_synthetic.py --out data/synth_facil --variation 0.5 --noise 0.05
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# se dibuja a 4x y se reduce: así los trazos salen suavizados
SUPERSAMPLE = 4


def random_glyph(rng: np.random.Generator) -> list[np.ndarray]:
    """Un glifo: de 2 a 4 trazos, cada uno una polilínea de 2 a 4 puntos en [-1, 1]²."""
    return [rng.uniform(-0.8, 0.8, size=(rng.integers(2, 5), 2)) for _ in range(rng.integers(2, 5))]


def render(
    glyph: list[np.ndarray],
    size: int,
    variation: float,
    noise: float,
    rng: np.random.Generator,
) -> Image.Image:
    angle = rng.normal(0, math.radians(15) * variation)
    scale = 1 + rng.normal(0, 0.1 * variation)
    shift = rng.normal(0, 0.08 * variation, size=2)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])

    big = size * SUPERSAMPLE
    canvas = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(canvas)
    width = max(1, round(big * 0.07 * (1 + rng.normal(0, 0.2 * variation))))
    for stroke in glyph:
        # el temblor mueve cada punto por separado: deforma el glifo, no solo lo desplaza
        points = stroke + rng.normal(0, 0.05 * variation, size=stroke.shape)
        points = (points @ rotation.T) * scale + shift
        # de [-1, 1] a píxeles, con margen para que la rotación no corte el glifo
        pixels = (points * 0.8 + 1) / 2 * big
        draw.line([tuple(p) for p in pixels], fill=255, width=width, joint="curve")

    image = np.asarray(canvas.resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    if noise > 0:
        image = image + rng.normal(0, noise * 255, size=image.shape)
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--n-identities", type=int, default=200)
    parser.add_argument("--n-per-identity", type=int, default=20)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--variation", type=float, default=1.0, help="0 = sin variación, 1 = mucha")
    parser.add_argument("--noise", type=float, default=0.15, help="desviación del ruido, como fracción de 255")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for k in range(args.n_identities):
        glyph = random_glyph(rng)
        folder = args.out / f"id_{k:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        for n in range(args.n_per_identity):
            render(glyph, args.size, args.variation, args.noise, rng).save(folder / f"{n:03d}.png")

    total = args.n_identities * args.n_per_identity
    print(f"{args.n_identities} identidades · {total} imágenes en {args.out}")


if __name__ == "__main__":
    main()
