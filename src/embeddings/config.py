"""Hiperparámetros del entrenamiento, cargados desde un YAML."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    # datos
    data_root: str = "data/synthetic"
    test_frac: float = 0.3
    # si se da, estas identidades van a prueba y se ignora test_frac (MNIST: 8 y 9)
    test_ids: list[str] | None = None
    gallery_frac: float = 0.5
    min_images: int = 2
    image_size: int = 32
    grayscale: bool = True

    # modelo
    backbone: str = "small_cnn"
    pretrained: bool = True
    embedding_dim: int = 128

    # pérdida
    arcface_s: float = 30.0
    arcface_m: float = 0.4

    # entrenamiento
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 5e-4
    num_workers: int = 2
    seed: int = 0
    device: str = "auto"
    out_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict | None = None) -> "Config":
        with open(path) as f:
            values = yaml.safe_load(f) or {}
        values.update(overrides or {})
        # una clave mal escrita en el YAML truena aquí con TypeError, no se ignora
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
