"""Loop de entrenamiento: datos → modelo → ArcFace → evaluación por época."""

import csv
import json
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import IdentityImageDataset, build_transforms, make_open_set_split
from .evaluate import embed_all, evaluate
from .losses import ArcFace
from .models import EmbeddingModel


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def validate(model, criterion, dataset, labels: torch.Tensor, device) -> tuple[float, float]:
    """Pérdida y exactitud de `val` contra los centros de ArcFace (las filas de W).

    Son fotos nuevas de identidades conocidas: el número sale optimista y solo
    sirve para comparar épocas entre sí.
    """
    z = embed_all(model, dataset, device).to(device)
    labels = labels.to(device)
    centers = F.normalize(criterion.W, dim=1)
    accuracy = ((z @ centers.T).argmax(dim=1) == labels).float().mean().item()
    return criterion(z, labels).item(), accuracy


def _describe(name: str, items: list) -> str:
    n_ids = len({identity for _, identity in items})
    return f"{name:<8} {n_ids:>6} identidades · {len(items):>7} imágenes"


def train(cfg: Config) -> dict[str, float]:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    split = make_open_set_split(
        Path(cfg.data_root),
        test_frac=cfg.test_frac,
        test_ids=cfg.test_ids,
        gallery_frac=cfg.gallery_frac,
        val_frac=cfg.val_frac,
        min_images=cfg.min_images,
        seed=cfg.seed,
    )
    print(_describe("train", split.train))
    if split.val:
        print(_describe("val", split.val))
    print(_describe("galería", split.gallery))
    print(_describe("queries", split.query))

    leaked = {i for _, i in split.train} & {i for _, i in split.gallery + split.query}
    if leaked:
        # no se detiene: medir la fuga a propósito es un experimento válido
        print(
            f"\n⚠  FUGA: {len(leaked)} identidades están en train y en prueba. "
            "Estos números NO miden generalización.\n"
        )

    train_tf, eval_tf = build_transforms(cfg.image_size, cfg.grayscale)
    train_ds = IdentityImageDataset(split.train, train_tf, cfg.grayscale)
    gallery_ds = IdentityImageDataset(split.gallery, eval_tf, cfg.grayscale)
    query_ds = IdentityImageDataset(split.query, eval_tf, cfg.grayscale)
    val_ds = IdentityImageDataset(split.val, eval_tf, cfg.grayscale) if split.val else None
    if val_ds:
        # el mismo entero de clase que en train: la posición del nombre ordenado
        class_of = {name: k for k, name in enumerate(sorted(set(train_ds.identities)))}
        val_labels = torch.tensor([class_of[name] for name in val_ds.identities])

    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        # BatchNorm truena con un batch de tamaño 1: mejor tirar el último si queda incompleto
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    if len(loader) == 0:
        raise ValueError(
            f"El entrenamiento tiene {len(train_ds)} imágenes, menos que batch_size={cfg.batch_size}. "
            "Bajen batch_size en el YAML."
        )

    model = EmbeddingModel(
        cfg.backbone,
        cfg.embedding_dim,
        pretrained=cfg.pretrained,
        in_channels=1 if cfg.grayscale else 3,
    ).to(device)
    # una fila de W por identidad de ENTRENAMIENTO; se descarta al terminar
    criterion = ArcFace(
        cfg.embedding_dim,
        train_ds.num_classes,
        s=cfg.arcface_s,
        m=cfg.arcface_m,
    ).to(device)

    # los centros de clase de ArcFace también se entrenan: si se quedan fuera del
    # optimizador, W se congela en su inicialización y el modelo aprende contra
    # centros al azar
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(loader))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.out_dir) / f"{Path(cfg.data_root).name}_{cfg.backbone}_{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), allow_unicode=True, sort_keys=False))
    print(f"\ndevice {device} · {cfg.epochs} épocas · {len(loader)} pasos por época · salida en {run_dir}\n")

    best_key, best_epoch = None, None
    with open(run_dir / "metrics.csv", "w", newline="") as f:
        writer = None
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            start = time.time()
            total = 0.0
            for images, labels in tqdm(loader, desc=f"época {epoch}/{cfg.epochs}", leave=False):
                images, labels = images.to(device), labels.to(device)
                loss = criterion(model(images), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total += loss.item()

            row = {"epoch": epoch, "loss": total / len(loader)}
            if val_ds:
                row["val_loss"], row["val_acc"] = validate(model, criterion, val_ds, val_labels, device)
                # empates de exactitud (frecuentes cuando satura): gana la menor pérdida
                key = (row["val_acc"], -row["val_loss"])
            else:
                # sin validación no hay con qué elegir: se queda la última época
                key = (epoch,)
            row["lr"] = scheduler.get_last_lr()[0]
            row["seconds"] = round(time.time() - start, 1)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            f.flush()

            line = f"época {epoch:>3}  loss {row['loss']:.3f}"
            if val_ds:
                line += f"  val_loss {row['val_loss']:.3f}  val_acc {row['val_acc']:.3f}"
            print(f"{line}  ({row['seconds']}s)")

            if best_key is None or key > best_key:
                best_key, best_epoch = key, epoch
                # solo el modelo: el head de ArcFace ya no sirve después de entrenar
                torch.save({"model": model.state_dict(), "config": asdict(cfg)}, run_dir / "best.pt")

    # prueba se mide una sola vez, con la época que eligió validación. Medirla en
    # cada época y quedarse con la mejor es usar prueba para decidir: una fuga
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model"])
    metrics = evaluate(model, gallery_ds, query_ds, device)
    (run_dir / "test.json").write_text(json.dumps({"epoch": best_epoch, **metrics}, indent=2))
    chosen_by = "validación" if val_ds else "última época"
    print(
        f"\nmejor época {best_epoch} ({chosen_by}) · prueba, evaluada una sola vez:  "
        f"R@1 {metrics['recall@1']:.3f}  R@5 {metrics['recall@5']:.3f}  "
        f"mAP {metrics['mAP']:.3f}  TAR@FAR=1e-2 {metrics['tar@far=1e-2']:.3f}"
    )
    print(f"checkpoint en {run_dir / 'best.pt'}")
    return {"epoch": best_epoch, **metrics}
