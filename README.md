# Embeddings para imágenes — ejercicio

Curso de embeddings, Maestría en Ciencia de Datos, ITAM.

Casi todo el código ya está hecho. Lo que falta es lo que toca los datos, y ahí
es donde se gana el ejercicio.

## Instalación

Python 3.10 o más reciente.

```bash
git clone git@github.com:felipegerard/intro-to-embeddings.git
cd intro-to-embeddings
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Primero validen el entorno. No necesita datos y tarda segundos:

```bash
python scripts/smoke_test.py
```

Tiene que terminar con `✓ Todo en orden`. Si van a usar un backbone
preentrenado, corran también `python scripts/smoke_test.py --backbone resnet18`,
que además descarga los pesos.

## Qué implementan

Todo está en `src/embeddings/data.py`. Lean los docstrings: dicen exactamente
qué hace cada pieza y dónde están las trampas.

| Pieza | Qué hace |
|---|---|
| `make_open_set_split` | Reparte **identidades** entre entrenamiento y prueba, y dentro de prueba reparte imágenes entre galería y queries |
| `IdentityImageDataset` | Carga cada imagen, le asigna su entero de clase y expone `paths`, `identities` y `num_classes` |
| `build_transforms` | Augmentations de entrenamiento y un preprocesamiento determinista para evaluar |

Los tests son su lista de pendientes:

```bash
pytest
```

Hoy fallan `tests/test_split.py` y `tests/test_dataset.py`. Cuando pasen todos,
esta parte está lista. El resto de los tests ya pasa: cubre el código que se les
da.

Lo que **no** se toca: `src/embeddings/evaluate.py`. Es idéntico para todos los
equipos, para que los números se puedan comparar. Si creen que tiene un bug,
avisen.

## Los datos

Una carpeta por identidad. El nombre de la carpeta es la identidad:

```
data/<dataset>/<identidad>/<imagen>.png
```

**Todos corren el sintético común**: 200 glifos generados con semilla fija,
idénticos en todas las máquinas. Es la única columna comparable entre equipos,
así que no cambien los argumentos de `make_synthetic.py` ni la parte de datos de
`configs/synthetic.yaml`.

```bash
python scripts/make_synthetic.py
python scripts/run_train.py --config configs/synthetic.yaml
```

**Y un segundo dataset, a su elección:**

| Dataset | Cómo se parte en open-set |
|---|---|
| Omniglot | Se reservan alfabetos completos para prueba |
| MNIST (`export_mnist.py`) | 0–7 para entrenar, 8 y 9 para evaluar (`configs/mnist.yaml`) |
| Sintético propio | El mismo generador con otros argumentos: cuántas identidades, cuánta variación, cuánto ruido |
| Fashion-MNIST | Unas prendas para entrenar, otras para evaluar |
| CIFAR-100 | 80 clases para entrenar, 20 para evaluar |
| CUB-200-2011 | Aves 1–100 para entrenar, 101–200 para evaluar. Pide GPU |

Pueden traer otro si pueden definir un split por identidad sin fuga. El script
que convierte su dataset a carpeta-por-identidad es parte del entregable.

MNIST viene listo, pero ojo: es tan fácil que el vecino más cercano en píxeles
crudos ya da un Recall@1 de 0.97. Si lo eligen, su análisis tiene que explicar
qué aporta su modelo sobre eso.

```bash
python scripts/export_mnist.py
python scripts/run_train.py --config configs/mnist.yaml
```

## Entrenar

```bash
python scripts/run_train.py --config configs/synthetic.yaml
python scripts/run_train.py --config configs/synthetic.yaml --set backbone=resnet18 --set epochs=5
```

- `--set CLAVE=VALOR` sobreescribe cualquier clave del YAML; las opciones están
  en `src/embeddings/config.py`.
- `--device` fuerza `cpu`, `mps` o `cuda`. Por defecto se detecta solo.
- Backbones: `small_cnn` (desde cero, corre en CPU), `resnet18` y
  `mobilenet_v3_small` (preentrenados).

Cada corrida escribe en `runs/<nombre>/` la configuración, `metrics.csv` con las
métricas por época y `best.pt` con el mejor modelo según Recall@1.

## La línea base: píxeles crudos

Antes de presumir un número, midan el piso: el vecino más cercano comparando las
imágenes píxel a píxel, sin ningún modelo. Mismo split, misma evaluación.

```bash
python scripts/baseline_pixels.py --config configs/synthetic.yaml
```

Si su modelo no le gana con holgura, no aprendió nada que sirva. Es obligatorio
reportarla para sus dos datasets.

## Puntos extra

Máximo 2, uno por cada cosa:

- Triplet loss con batch-hard (`TripletLoss` en `losses.py`), comparada contra ArcFace
- CosFace
- Sampler balanceado por clase
- Fine-tuning por capas
- Ablación del head y de la dimensión del embedding
- Baseline DINOv2 congelado, sin entrenar nada
- Índice FAISS o HNSW, con latencia y recall contra fuerza bruta

## Entregables

- El repo con `data.py` completo y el script que convierte su dataset.
- La salida de la evaluación y de la línea base de píxeles, para el sintético
  común y para su dataset.
- Una presentación de 20 minutos en la sesión 3.

Suban el repo la noche anterior a la sesión 3.

## Calificación

- **50% resultado técnico**: que corra, que el split sea correcto, que los
  números sean creíbles.
- **50% presentación**: qué probaron, qué falló y qué concluyen.
- Hasta **+2** por extras.

Un modelo con malos números y un buen análisis saca 8. Lo que se evalúa es el
criterio, no el recall.

## Licencia

Código bajo MIT ([LICENSE](LICENSE)). Material didáctico bajo CC BY-NC-ND 4.0
([LICENSE-MATERIAL.md](LICENSE-MATERIAL.md)).
