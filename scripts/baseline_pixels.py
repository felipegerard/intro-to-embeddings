"""Línea base sin modelo: el vecino más cercano en píxeles crudos.

Usa el mismo split, el mismo preprocesamiento de evaluación y la misma
evaluación que el entrenamiento, pero en vez de un embedding aprendido compara
las imágenes píxel a píxel. Es el piso: si su modelo no le gana con holgura, no
aprendió nada que sirva.

    python scripts/baseline_pixels.py --config configs/synthetic.yaml
"""

import argparse
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

from embeddings.config import Config
from embeddings.data import IdentityImageDataset, build_transforms, make_open_set_split
from embeddings.evaluate import evaluate


class RawPixels(nn.Module):
    """El «embedding» es la imagen aplanada y normalizada: nada aprendido."""

    def forward(self, x):
        return F.normalize(x.flatten(1), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    split = make_open_set_split(
        Path(cfg.data_root),
        test_frac=cfg.test_frac,
        test_ids=cfg.test_ids,
        gallery_frac=cfg.gallery_frac,
        min_images=cfg.min_images,
        seed=cfg.seed,
    )
    _, eval_tf = build_transforms(cfg.image_size, cfg.grayscale)
    gallery = IdentityImageDataset(split.gallery, eval_tf, cfg.grayscale)
    query = IdentityImageDataset(split.query, eval_tf, cfg.grayscale)

    m = evaluate(RawPixels(), gallery, query, "cpu")
    print(
        f"píxeles crudos · {Path(cfg.data_root).name}  "
        f"R@1 {m['recall@1']:.3f}  R@5 {m['recall@5']:.3f}  "
        f"mAP {m['mAP']:.3f}  TAR@FAR=1e-2 {m['tar@far=1e-2']:.3f}"
    )


if __name__ == "__main__":
    main()
