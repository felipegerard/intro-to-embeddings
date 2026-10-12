"""Contrato de `IdentityImageDataset` y `build_transforms`. Hoy fallan."""

import torch

from embeddings.data import IdentityImageDataset, build_transforms


def all_items(root):
    # sin pasar por el split: así se pueden implementar en cualquier orden
    return [(path, path.parent.name) for path in sorted(root.rglob("*.png"))]


def test_length_and_item(tree):
    items = all_items(tree)
    _, eval_tf = build_transforms(image_size=24, grayscale=True)
    ds = IdentityImageDataset(items, eval_tf, grayscale=True)
    assert len(ds) == len(items)
    image, label = ds[0]
    assert isinstance(image, torch.Tensor) and image.shape == (1, 24, 24)
    assert isinstance(label, int) and 0 <= label < ds.num_classes


def test_attributes_used_by_evaluation(tree):
    items = all_items(tree)
    _, eval_tf = build_transforms(image_size=24)
    ds = IdentityImageDataset(items, eval_tf)
    assert ds.paths == [str(path) for path, _ in items]
    assert ds.identities == [identity for _, identity in items]
    assert ds.num_classes == len(set(ds.identities))


def test_one_label_per_identity(tree):
    items = all_items(tree)
    _, eval_tf = build_transforms(image_size=24)
    ds = IdentityImageDataset(items, eval_tf)
    labels = {}
    for k in range(len(ds)):
        labels.setdefault(ds.identities[k], set()).add(ds[k][1])
    assert all(len(found) == 1 for found in labels.values())
    assert {found.pop() for found in labels.values()} == set(range(ds.num_classes))


def test_rgb_mode(tree):
    _, eval_tf = build_transforms(image_size=24, grayscale=False)
    image, _ = IdentityImageDataset(all_items(tree), eval_tf, grayscale=False)[0]
    assert image.shape == (3, 24, 24)


def test_eval_transform_is_deterministic(tree):
    _, eval_tf = build_transforms(image_size=24)
    ds = IdentityImageDataset(all_items(tree), eval_tf)
    assert torch.equal(ds[0][0], ds[0][0])


def test_train_transform_keeps_shape(tree):
    train_tf, _ = build_transforms(image_size=24)
    image, _ = IdentityImageDataset(all_items(tree), train_tf)[0]
    assert image.shape == (1, 24, 24)


def test_label_is_position_of_sorted_name(tree):
    # train.py etiqueta la validación con esta misma regla
    items = all_items(tree)
    _, eval_tf = build_transforms(image_size=24)
    ds = IdentityImageDataset(items, eval_tf)
    names = sorted(set(ds.identities))
    for k in (0, len(ds) // 2, len(ds) - 1):
        assert ds[k][1] == names.index(ds.identities[k])
