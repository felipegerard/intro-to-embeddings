"""Contrato de `make_open_set_split`. Hoy fallan; cuando pasen todos, el split está bien."""

from embeddings.data import make_open_set_split


def ids(items):
    return {identity for _, identity in items}


def test_no_identity_leak(tree):
    # la regla que no se puede romper: ninguna identidad en train y en prueba a la vez
    split = make_open_set_split(tree, test_frac=0.3, seed=0)
    assert ids(split.train) & (ids(split.gallery) | ids(split.query)) == set()


def test_every_query_identity_is_in_gallery(tree):
    split = make_open_set_split(tree, test_frac=0.3, seed=0)
    assert ids(split.query) == ids(split.gallery)


def test_each_image_used_once(tree):
    split = make_open_set_split(tree, test_frac=0.3, seed=0)
    paths = [path for path, _ in split.train + split.gallery + split.query]
    assert len(paths) == len(set(paths))


def test_drops_identities_below_min_images(tree):
    split = make_open_set_split(tree, test_frac=0.3, min_images=2, seed=0)
    seen = ids(split.train) | ids(split.gallery) | ids(split.query)
    assert not any(name.startswith("solo_") for name in seen)
    assert len(seen) == 20


def test_test_frac_counts_identities_not_images(tree):
    split = make_open_set_split(tree, test_frac=0.3, seed=0)
    assert len(ids(split.gallery)) == 6


def test_explicit_test_ids(tree):
    split = make_open_set_split(tree, test_ids=["id_03", "id_07"], seed=0)
    assert ids(split.gallery) == {"id_03", "id_07"}
    assert {"id_03", "id_07"}.isdisjoint(ids(split.train))


def test_test_ids_given_as_numbers(make_tree):
    # el YAML convierte `8` en entero, pero las carpetas se llaman "8"
    root = make_tree({str(d): 3 for d in range(10)})
    split = make_open_set_split(root, test_ids=[8, 9], seed=0)
    assert ids(split.gallery) == {"8", "9"}


def test_gallery_frac(tree):
    # id_06 tiene 8 imágenes: la mitad a galería, la mitad a queries
    split = make_open_set_split(tree, test_ids=["id_06"], gallery_frac=0.5, seed=0)
    assert len(split.gallery) == 4 and len(split.query) == 4


def test_deterministic(tree):
    assert make_open_set_split(tree, seed=0) == make_open_set_split(tree, seed=0)


def test_seed_changes_split(tree):
    a = make_open_set_split(tree, seed=0)
    b = make_open_set_split(tree, seed=1)
    assert ids(a.gallery) != ids(b.gallery)
