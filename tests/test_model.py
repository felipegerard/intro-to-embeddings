"""El código dado: modelo y ArcFace. Estos tests pasan desde el día uno."""

import pytest
import torch
import torch.nn.functional as F

from embeddings.losses import ArcFace
from embeddings.models import BACKBONES, EmbeddingModel


@pytest.mark.parametrize("backbone", BACKBONES)
def test_embeddings_are_normalized(backbone):
    # entrada en gris a propósito: los preentrenados la tienen que copiar a 3 canales
    model = EmbeddingModel(backbone, embedding_dim=32, pretrained=False, in_channels=1).eval()
    z = model(torch.randn(4, 1, 32, 32))
    assert z.shape == (4, 32)
    assert torch.allclose(z.norm(dim=1), torch.ones(4), atol=1e-5)


def test_arcface_is_finite_and_trains_its_centers():
    torch.manual_seed(0)
    criterion = ArcFace(16, n_classes=5)
    z = F.normalize(torch.randn(8, 16), dim=1)
    loss = criterion(z, torch.arange(8) % 5)
    loss.backward()
    assert torch.isfinite(loss)
    assert criterion.W.grad is not None and criterion.W.grad.abs().sum() > 0


def test_margin_makes_the_task_harder():
    # con margen, la clase correcta tiene que ganar por más: la pérdida sube
    torch.manual_seed(0)
    z = F.normalize(torch.randn(8, 16), dim=1)
    labels = torch.arange(8) % 5
    no_margin = ArcFace(16, 5, m=0.0)
    with_margin = ArcFace(16, 5, m=0.4)
    with_margin.W.data.copy_(no_margin.W.data)
    assert with_margin(z, labels) > no_margin(z, labels)


def test_arcface_survives_perfect_alignment():
    # z idéntico a su centro ⇒ cos = 1 exacto; sin el clamp esto da NaN
    criterion = ArcFace(16, n_classes=3)
    z = F.normalize(criterion.W.detach()[:1].clone(), dim=1).requires_grad_()
    loss = criterion(z, torch.tensor([0]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
