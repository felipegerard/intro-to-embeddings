"""Exporta MNIST al formato del curso: una carpeta por dígito.

Es el dataset común de todos los equipos, así que no cambien `--per-class`:
mismas imágenes para todos. Se entrena con los dígitos 0–7 y se evalúa con 8 y
9 (ver `configs/mnist.yaml`).

    python scripts/export_mnist.py
"""

import argparse
from collections import defaultdict
from pathlib import Path

from torchvision.datasets import MNIST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/mnist"))
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--download-dir", type=Path, default=Path("data/_downloads"))
    args = parser.parse_args()

    mnist = MNIST(args.download_dir, train=True, download=True)
    counts = defaultdict(int)
    # las primeras `per_class` de cada dígito, en el orden del archivo: el
    # resultado es idéntico en todas las máquinas
    for image, digit in mnist:
        if counts[digit] >= args.per_class:
            continue
        folder = args.out / str(digit)
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / f"{counts[digit]:05d}.png")
        counts[digit] += 1
        if len(counts) == 10 and min(counts.values()) >= args.per_class:
            break

    print(f"{sum(counts.values())} imágenes en {args.out} ({args.per_class} por dígito)")


if __name__ == "__main__":
    main()
