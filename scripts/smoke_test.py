"""Valida el entorno: torch, el device, el modelo y ArcFace. No necesita datos.

Entrena unos pasos sobre un batch fijo de imágenes al azar y revisa que la
pérdida sea finita, que baje, y que los embeddings salgan normalizados. Es lo
que se corre en la sesión 2 antes de irse.

    python scripts/smoke_test.py
    python scripts/smoke_test.py --backbone resnet18   # además descarga los pesos
"""

import argparse
import platform

import torch

from embeddings.losses import ArcFace
from embeddings.models import BACKBONES, EmbeddingModel
from embeddings.train import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", default="small_cnn", choices=BACKBONES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"python {platform.python_version()} · torch {torch.__version__} · device {device}")

    torch.manual_seed(0)
    n_classes, batch, size, dim = 10, 32, 32, 64
    channels = 1 if args.backbone == "small_cnn" else 3
    model = EmbeddingModel(args.backbone, embedding_dim=dim, in_channels=channels).to(device)
    criterion = ArcFace(dim, n_classes).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3)

    # siempre el mismo batch: si todo funciona, la red lo memoriza y la pérdida baja
    images = torch.randn(batch, channels, size, size, device=device)
    labels = torch.arange(batch, device=device) % n_classes

    losses = []
    model.train()
    for step in range(1, args.steps + 1):
        z = model(images)
        loss = criterion(z, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise SystemExit(f"✗ La pérdida no es finita en el paso {step}.")
        losses.append(loss.item())
        if step == 1 or step % 5 == 0:
            print(f"paso {step:>3}  loss {losses[-1]:.3f}")

    norms = z.detach().norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        raise SystemExit("✗ Los embeddings no tienen norma 1.")
    if losses[-1] >= losses[0]:
        raise SystemExit(f"✗ La pérdida no bajó: {losses[0]:.3f} → {losses[-1]:.3f}.")
    print(f"\n✓ Todo en orden: la pérdida bajó de {losses[0]:.3f} a {losses[-1]:.3f} y los embeddings tienen norma 1.")


if __name__ == "__main__":
    main()
