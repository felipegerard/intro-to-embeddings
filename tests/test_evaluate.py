"""La evaluación congelada, en casos donde sabemos la respuesta."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings.evaluate import evaluate


class VectorDataset(torch.utils.data.Dataset):
    """Un dataset cuyas «imágenes» ya son vectores: prueba la evaluación sola."""

    def __init__(self, vectors, identities, paths):
        self.vectors, self.identities, self.paths = vectors, identities, paths

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, i):
        return self.vectors[i], 0


class Normalize(nn.Module):
    def forward(self, x):
        return F.normalize(x, dim=1)


def clusters(n_ids, per_id, spread, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_ids, 16, generator=g)
    vectors = centers.repeat_interleave(per_id, 0) + spread * torch.randn(n_ids * per_id, 16, generator=g)
    identities = [f"id_{k}" for k in range(n_ids) for _ in range(per_id)]
    return vectors, identities


def test_separated_clusters_give_perfect_scores():
    v, ids = clusters(10, 6, spread=0.01)
    gallery = VectorDataset(v[0::2], ids[0::2], [f"g{k}" for k in range(30)])
    query = VectorDataset(v[1::2], ids[1::2], [f"q{k}" for k in range(30)])
    m = evaluate(Normalize(), gallery, query, "cpu")
    assert m["recall@1"] == 1.0 and m["mAP"] == 1.0 and m["tar@far=1e-2"] == 1.0


def test_query_never_matches_itself():
    # galería = queries, con identidades que no tienen relación con los
    # vectores: sin el enmascarado cada query se encontraría a sí misma y el
    # recall daría 1.0
    v, _ = clusters(10, 6, spread=0.01)
    unrelated = [f"id_{k % 10}" for k in range(60)]
    ds = VectorDataset(v, unrelated, [f"img{k}" for k in range(60)])
    assert evaluate(Normalize(), ds, ds, "cpu")["recall@1"] < 0.5


def test_queries_without_a_gallery_match_are_excluded():
    v, ids = clusters(4, 4, spread=0.01)
    # id_3 solo aparece en queries: no puede acertar, así que no debe contar
    g_idx = [k for k in range(16) if ids[k] != "id_3" and k % 2 == 0]
    q_idx = [k for k in range(16) if k % 2 == 1]
    gallery = VectorDataset(v[g_idx], [ids[k] for k in g_idx], [f"p{k}" for k in g_idx])
    query = VectorDataset(v[q_idx], [ids[k] for k in q_idx], [f"p{k}" for k in q_idx])
    m = evaluate(Normalize(), gallery, query, "cpu")
    assert m["n_queries"] == 6 and m["recall@1"] == 1.0
