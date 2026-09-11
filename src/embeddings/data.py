"""Los datos. ESTE ARCHIVO LO IMPLEMENTAN USTEDES.

Tres piezas:

1. `make_open_set_split` — decide qué imágenes van a entrenamiento, a galería
   y a queries. Es donde se gana o se pierde el ejercicio.
2. `IdentityImageDataset` — carga una imagen y le pone su etiqueta.
3. `build_transforms` — las augmentations de entrenamiento y el
   preprocesamiento de evaluación.

Los tests de `tests/test_split.py` y `tests/test_dataset.py` son su lista de
pendientes: hoy fallan todos, y cuando pasen todos esta parte está lista.

Formato esperado en disco, una carpeta por identidad:

    data/<dataset>/<identidad>/<imagen>.png

El nombre de la carpeta ES la identidad. No hace falta ningún CSV.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset

# (ruta de la imagen, nombre de su identidad)
Item = tuple[Path, str]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class Split:
    train: list[Item]
    gallery: list[Item]
    query: list[Item]


def make_open_set_split(
    root: Path,
    test_frac: float = 0.3,
    test_ids: list[str] | None = None,
    gallery_frac: float = 0.5,
    min_images: int = 2,
    seed: int = 0,
) -> Split:
    """Reparte IDENTIDADES entre entrenamiento y prueba; luego, dentro de
    prueba, reparte IMÁGENES entre galería y queries.

    Pasos:

    1. Recorrer `root`: cada subcarpeta es una identidad, y sus archivos con
       extensión en `IMAGE_EXTENSIONS` son sus imágenes.
    2. Descartar las identidades con menos de `min_images` imágenes. Con una
       sola imagen no hay par positivo que aprender, y en prueba no se puede
       poner a la vez en galería y en queries.
    3. Elegir las identidades de prueba:
       - si `test_ids` viene dado, son exactamente esas (MNIST usa `["8", "9"]`).
         Compárenlas como texto: el YAML puede entregar `8` como entero;
       - si no, una fracción `test_frac` de las identidades, al azar con `seed`.
       Todas las imágenes de una identidad de entrenamiento van a `train`.
    4. Para cada identidad de prueba, mandar ~`gallery_frac` de sus imágenes a
       `gallery` y el resto a `query`, dejando al menos una de cada lado.

    La regla que no se puede romper: **ninguna identidad aparece en
    entrenamiento y en prueba a la vez.** Repartir imágenes primero e
    identidades después es exactamente la fuga que vimos en la sesión 1.

    Tiene que ser determinista: la misma `seed` produce el mismo split. Ordenen
    carpetas y archivos antes de barajar, porque el orden de `iterdir()` depende
    del sistema operativo.

    Devuelve un `Split` con tres listas de `(ruta, identidad)`.
    """
    raise NotImplementedError(
        "make_open_set_split: esta función la implementan ustedes. "
        "Lean su docstring y la sección «Qué implementan» del README."
    )


class IdentityImageDataset(Dataset):
    """Dataset de imágenes etiquetadas por identidad.

    Recibe una lista de `(ruta, identidad)`, como las de `Split`, y una
    transformación. Tiene que:

    - asignar a cada identidad un entero de clase, de 0 a `num_classes - 1`,
      ordenando los nombres para que sea determinista. ArcFace necesita esos
      enteros, y solo existen durante el entrenamiento;
    - en `__getitem__(i)`: abrir la imagen con PIL, convertirla a `"L"` (gris)
      o `"RGB"` según `grayscale`, aplicarle `transform` y devolver
      `(tensor, entero_de_clase)`;
    - exponer tres atributos que usa `evaluate.py`:
        `paths`        lista de rutas como texto, una por imagen, en orden;
        `identities`   lista de nombres de identidad, una por imagen, en orden;
        `num_classes`  número de identidades distintas.

    La evaluación compara `identities` y no los enteros de clase: la galería y
    las queries son datasets separados y cada uno numera a su manera.
    """

    def __init__(
        self,
        items: list[Item],
        transform: Callable,
        grayscale: bool = True,
    ):
        raise NotImplementedError(
            "IdentityImageDataset: esta clase la implementan ustedes. "
            "Lean su docstring y la sección «Qué implementan» del README."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        raise NotImplementedError


def build_transforms(
    image_size: int,
    grayscale: bool = True,
) -> tuple[Callable, Callable]:
    """Devuelve `(transform_de_entrenamiento, transform_de_evaluación)`.

    Las dos reciben una imagen PIL y devuelven un tensor
    `(C, image_size, image_size)`, con C = 1 si `grayscale` y 3 si no.

    - **Entrenamiento**: augmentations que preserven la identidad. Piensen en
      su dominio: voltear horizontalmente una «b» la convierte en «d», y pasar a
      gris un producto cuyo color es su identidad la destruye.
    - **Evaluación**: nada aleatorio. Solo redimensionar y convertir a tensor (y
      normalizar, si normalizan en entrenamiento). La misma imagen tiene que dar
      siempre el mismo tensor, o los números no son reproducibles.

    Con un backbone preentrenado (`resnet18`, `mobilenet_v3_small`) y en color,
    conviene normalizar con la media y la desviación estándar de ImageNet:
    `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
    """
    raise NotImplementedError(
        "build_transforms: esta función la implementan ustedes. "
        "Lean su docstring y la sección «Qué implementan» del README."
    )
