"""Evaluación del curso. NO MODIFICAR.

Este archivo es idéntico para todos los equipos: así los números de la sesión 3
se pueden poner en la misma tabla. Si creen que tiene un bug, avisen y se
arregla para todos.

Todas las métricas comparan IDENTIDADES (el nombre de la carpeta), no los
enteros de clase, así que no importa cómo numere cada dataset sus clases.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def embed_all(model, dataset, device, batch_size: int = 256) -> torch.Tensor:
    """Embeddings de todo el dataset, en orden y en CPU."""
    model.eval()
    # num_workers=0: la evaluación no es cuello de botella y así es determinista
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    chunks = [model(images.to(device)).float().cpu() for images, _ in loader]
    return torch.cat(chunks)


def _identity_ids(*id_lists: list[str]) -> list[torch.Tensor]:
    # el mismo entero para la misma identidad en galería y en queries
    vocab = {name: k for k, name in enumerate(sorted(set().union(*id_lists)))}
    return [torch.tensor([vocab[name] for name in ids]) for ids in id_lists]


def mean_average_precision(hits: torch.Tensor) -> float:
    """mAP a partir de la matriz de aciertos ya ordenada, `(n_queries, n_gallery)`."""
    ranks = torch.arange(1, hits.size(1) + 1)
    precision = hits.cumsum(dim=1) / ranks
    ap = (precision * hits).sum(dim=1) / hits.sum(dim=1)
    return ap.mean().item()


def tar_at_far(
    z: torch.Tensor,
    ids: torch.Tensor,
    far: float = 1e-2,
    max_samples: int = 3000,
    seed: int = 0,
) -> float:
    """TAR a un FAR dado, sobre todos los pares entre las muestras.

    El umbral se elige para que solo una fracción `far` de los pares de
    identidades distintas lo supere; el TAR es la fracción de pares de la
    misma identidad que sí lo superan.
    """
    if len(z) > max_samples:
        # submuestra fija para acotar la memoria: los pares crecen como n²
        g = torch.Generator().manual_seed(seed)
        keep = torch.randperm(len(z), generator=g)[:max_samples]
        z, ids = z[keep], ids[keep]
    i, j = torch.triu_indices(len(z), len(z), offset=1)
    pair_sim = (z @ z.T)[i, j]
    same = ids[i] == ids[j]
    if same.all() or not same.any():
        return float("nan")
    threshold = np.quantile(pair_sim[~same].numpy(), 1 - far)
    return (pair_sim[same] > threshold).float().mean().item()


def evaluate(model, gallery, query, device) -> dict[str, float]:
    """Recall@1, Recall@5, mAP y TAR@FAR=1e-2 de las queries contra la galería.

    `gallery` y `query` deben exponer `.paths` e `.identities` (listas con un
    elemento por imagen), como `IdentityImageDataset`.
    """
    zg = embed_all(model, gallery, device)
    zq = embed_all(model, query, device)
    yg, yq = _identity_ids(gallery.identities, query.identities)

    # todo normalizado ⇒ el producto punto ES el coseno
    sim = zq @ zg.T

    # una query nunca puede recuperarse a sí misma, aunque esté en la galería
    gallery_index = {path: k for k, path in enumerate(gallery.paths)}
    self_match = torch.zeros_like(sim, dtype=torch.bool)
    for q, path in enumerate(query.paths):
        if path in gallery_index:
            self_match[q, gallery_index[path]] = True
    sim[self_match] = -2

    order = sim.argsort(dim=1, descending=True)
    hits = (yg[order] == yq[:, None]) & ~self_match.gather(1, order)

    # una query cuya identidad no tiene ninguna otra imagen en la galería no
    # puede acertar nunca: se excluye en vez de contarla como error
    valid = hits.any(dim=1)
    hits = hits[valid]

    return {
        "recall@1": hits[:, :1].any(dim=1).float().mean().item(),
        "recall@5": hits[:, :5].any(dim=1).float().mean().item(),
        "mAP": mean_average_precision(hits.float()),
        "tar@far=1e-2": tar_at_far(zq, yq, far=1e-2),
        "n_queries": int(valid.sum()),
    }
