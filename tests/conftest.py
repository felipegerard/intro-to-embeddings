import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def make_tree(tmp_path):
    """Crea `<raíz>/<identidad>/<n>.png` con imágenes al azar."""

    def _make(counts: dict[str, int], size: int = 16):
        root = tmp_path / "data"
        rng = np.random.default_rng(0)
        for identity, n in counts.items():
            folder = root / identity
            folder.mkdir(parents=True)
            for k in range(n):
                pixels = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{k:03d}.png")
        return root

    return _make


@pytest.fixture
def tree(make_tree):
    # 20 identidades con entre 2 y 8 imágenes, más 3 que solo tienen una
    counts = {f"id_{k:02d}": 2 + k % 7 for k in range(20)}
    counts |= {f"solo_{k}": 1 for k in range(3)}
    return make_tree(counts)
