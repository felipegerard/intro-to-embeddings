# intro-to-embeddings

Curso de embeddings para imágenes — Maestría en Ciencia de Datos, ITAM.
Tres sesiones de 2 h. Los alumnos ya llevaron machine learning y/o deep learning.

Este archivo guarda las decisiones del curso. No repite lo que el código ya dice.

## Qué es un embedding aquí

`backbone + head → vector normalizado`, y el parecido se mide con coseno. El caso
de uso es **open-set**: clasificación donde pueden aparecer clases nuevas,
reconocimiento facial, búsqueda por similitud, deduplicación.

## Estructura del curso

| Sesión | Contenido |
|---|---|
| 1 · Teoría | Recap de redes (0:15), el problema open-set (0:20), arquitectura (0:20), receso (0:05), pérdidas (0:40), evaluación (0:20), producción (0:15) |
| 2 · Implementación | Recap de torch (0:20), recorrido del código (0:40), receso (0:05), los datos (0:30), el ejercicio (0:15), arranque en vivo (0:10) |
| 3 · Presentaciones | 4 equipos × 20 min (1:20), receso (0:05), comparación cruzada (0:20), cierre (0:15) |

Equipos de 3–4 personas, 3–4 equipos. Una semana para el ejercicio entre la
sesión 2 y la 3.

## Decisiones de diseño del ejercicio

### ArcFace, no triplet

La pérdida implementada y entregada es **ArcFace**. Triplet queda como punto
extra, no como base. Las razones, por si se quiere revisitar:

- Con ArcFace no hace falta muestreo PK ni minería de negativos: batches
  aleatorios bastan. Eso quita la mitad del código y casi todos los bugs.
- Converge estable, como un clasificador normal.
- Es lo que se usa hoy en producción. Enseñar triplet como método principal es
  enseñar algo ya superado.

Consecuencia: el sampler PK **salió** de la base del ejercicio. Sobrevive como
extra en su forma útil (sampler balanceado por clase, para datos
desbalanceados).

Triplet sí se cubre en teoría, ~10 min, porque la intuición geométrica de
ancla/positivo/negativo hace legible el margen angular, y porque se lo van a
topar en cualquier repo.

### Reparto: qué se les da y qué implementan

El ejercicio es sobre **datos**, no sobre modelado. Se les da casi todo el
código; lo que falta es lo que toca los datos.

| Archivo | Estado |
|---|---|
| `src/embeddings/data.py` | **Lo implementan ellos** |
| `src/embeddings/models.py` | Dado — backbones + head |
| `src/embeddings/losses.py` | Dado — ArcFace |
| `src/embeddings/train.py` | Dado — loop completo |
| `src/embeddings/config.py` | Dado |
| `src/embeddings/evaluate.py` | **Dado y congelado** |
| `scripts/make_synthetic.py` | Dado |
| `scripts/run_train.py` | Dado |
| `scripts/export_mnist.py` | Dado — MNIST idéntico para todos |
| `scripts/baseline_pixels.py` | Dado — el piso sin modelo |
| `scripts/smoke_test.py` | Dado — valida el entorno sin `data.py` |
| `tests/` | Dado — el contrato de `data.py`; su lista de pendientes |

Las tres piezas de `data.py`:

- `IdentityImageDataset` (~40 líneas)
- `make_open_set_split` (~35 líneas)
- `build_transforms` (~15 líneas)

Más el trabajo real: elegir dataset, convertirlo a carpeta-por-identidad,
mirarlo, limpiarlo, decidir augmentations.

### Solución de referencia

`_solucion/data.py` implementa las tres piezas y **no se versiona** (está en
`.gitignore`), porque el repo es público. Sirve para calificar y para verificar
el repo de punta a punta:

```bash
cp _solucion/data.py src/embeddings/data.py
pytest && python scripts/run_train.py --config configs/synthetic.yaml
git checkout src/embeddings/data.py
```

Junto a ella, sin versionar: `_solucion/acordeon.md` (acordeón del profesor: las
soluciones de todos los puntos extra y qué revisar al calificar),
`_solucion/extras.py` (esas soluciones en código) y `_solucion/probar_extras.py`
(las corre todas sobre el sintético común).

### La evaluación va congelada

`evaluate.py` no se toca. Recall@1, Recall@5, mAP y TAR@FAR. Si cada equipo mide
distinto, la tabla comparativa de la sesión 3 no significa nada. Si alguien
reporta un bug, se arregla para todos.

Detalle que importa: **una query nunca cuenta como su propio acierto**. Sin eso
todos los recall dan casi 1.0.

### Datos: una carpeta por identidad

```
data/<dataset>/<id>/<imagen>.jpg
```

El nombre de la carpeta es la identidad. No hace falta CSV. Distinción que se
repite todo el curso: **"perro" es una clase, "Firulais" es una identidad.**

El split reparte **identidades**, no imágenes. Ninguna identidad puede estar en
train y en test a la vez. Hay un test que falla si
`set(train_ids) & set(test_ids)` no está vacío.

Dentro de prueba, `gallery_frac` (0.5) de las imágenes de cada identidad van a
galería y el resto a queries. Las identidades con menos de `min_images` (2) se
descartan por completo: no sirven para entrenar, y en prueba no pueden estar a
la vez en galería y en queries. MNIST fija su split con `test_ids: ["8", "9"]`.

**Validación con identidades conocidas** (decidido el 2026-10-12): `val_frac`
(0.1) de las fotos de cada identidad de entrenamiento con 3 o más imágenes se
apartan para validación. `train.py` las compara contra los centros de ArcFace
(exactitud, con la pérdida para desempatar) y con eso elige `best.pt`. Prueba se
evalúa una sola vez, al final. Es el esquema de qface:

- Antes, `train.py` elegía la época por el R@1 de prueba: una fuga. Inflaba MNIST
  de 0.968 a 0.978.
- La validación closed-set sale optimista y no detecta sobreajuste open-set, pero
  solo ordena épocas; en nuestras corridas, la mejor y la última época de prueba
  difieren 0.00–0.01.
- No se parte en tres por identidad porque aquí las identidades son lo escaso
  (MNIST tiene 8 de entrenamiento). Con datos de sobra sería lo correcto; la
  lámina del split lo menciona.
- La validación usa un generador aparte: cambiar `val_frac` no mueve prueba.

### Datasets

Todos corren el **sintético común**: `make_synthetic.py` sin argumentos (200
identidades × 20 imágenes, `variation=1.0`, `noise=0.15`, seed 0). Es la columna
comparable de la sesión 3. Más un segundo dataset a su elección: Omniglot,
MNIST, sintético propio, Fashion-MNIST, CIFAR-100 o CUB-200-2011 (este pide GPU).
Pueden traer otro si pueden definir el split por identidad sin fuga.

**Por qué no MNIST** (decidido el 2026-09-10, después de medirlo): es tan fácil
que el vecino más cercano en píxeles crudos le gana en R@1 al modelo entrenado,
así que el número de un equipo no dice nada de su modelo.

| R@1, `small_cnn`, config por defecto | Píxeles crudos | Entrenado |
|---|---|---|
| MNIST 0–7 / 8–9 | 0.976 | 0.960 |
| MNIST 0–4 / 5–9 | 0.956 | 0.900 |
| Sintético común | 0.688 | 0.942 |

MNIST sí puede ser su segundo dataset (decidido el 2026-09-10), pero no la columna
comparable. Si lo eligen, tienen que explicar qué aporta su modelo sobre los píxeles.

**Línea base de píxeles, obligatoria** (decidido el 2026-09-10):
`baseline_pixels.py` evalúa el vecino más cercano en píxeles crudos con el mismo
split y la misma evaluación. Cada equipo la reporta para sus dos datasets. En la
tabla de la sesión 3 es una fila de referencia (≈ 0.69 de R@1 en el sintético
común) y una columna para el dataset propio.

### Backbones

`small_cnn` (28×28 en CPU), `resnet18` preentrenada, `mobilenet_v3_small`. El
head base se da funcionando; ampliarlo a 3–4 capas es extra, con ablación.

Nadie tiene GPU institucional. Se asume que al menos una persona por equipo
tiene una Mac con MPS o una PC gamer. Todo debe correr en CPU con pocas clases.

### Calificación

50% resultado técnico + 50% presentación. Un modelo con malos números y buen
análisis saca ~8; buenos números sin explicación, no. Máximo **2 puntos extra**,
uno por cosa (triplet, ArcFace sin `acos`, CosFace, sampler balanceado, fine-tuning por capas,
ablación de head/dimensión, baseline DINOv2 congelado, índice FAISS/HNSW).

## Convenciones de código

- Identificadores y nombres de variables **en inglés**; comentarios y docstrings
  **en español**.
- Español neutro latinoamericano. Nunca voseo. Acentos y ñ siempre, incluido en
  comentarios.
- Comentarios cortos y escasos, pero el público son alumnos que apenas conocen
  PyTorch: **sí vale explicar lo que a un experto le parecería obvio** cuando es
  una trampa real o un formato no evidente. Nada de narrar la línea siguiente.
- f-strings, no `.format()` ni `%`.
- Indentación multilínea estilo C: romper después del paréntesis, parámetros a un
  nivel, cerrar en la indentación original.
- Sin manejo de errores, perillas de configuración ni abstracciones que la tarea
  no pidió.

## Gráficas

El autor tiene deuteranomalía. En cualquier gráfica que produzca el código o el
material:

- Nunca distinguir series solo por tono, y nunca rojo contra verde.
- Paleta segura para daltonismo (Okabe-Ito). En el material, azul `#0072B2` es
  "positivo / misma identidad" y ámbar `#E69F00` es "negativo / distinta
  identidad", siempre con marcador distinto además del color.
- Líneas gruesas (~2–2.5 pt) y opacidad alta (~0.85).

## Material

`slides/embeddings.html` — un solo archivo autocontenido, 53 láminas, 16:9.
Navegación: `←` `→` navegar, `O` índice, `S` notas del ponente, `F` pantalla
completa. `Cmd+P` en Chrome saca el PDF.

Marca ITAM: verde `#006852` y `#004434`, logo blanco sobre fondo oscuro en la
cinta inferior. Bricolage Grotesque para display, IBM Plex Sans para cuerpo, IBM
Plex Mono para código.

## Licencias

- Código: **MIT** (`LICENSE`).
- Material didáctico: **CC BY-NC-ND 4.0** (`LICENSE-MATERIAL.md`). No se puede
  usar para dar clases de paga ni para capacitación en empresas sin permiso.
- El logo del ITAM es del ITAM. Los datasets conservan su propia licencia; este
  repo no los redistribuye.

## Git

- Commits en español con prefijo convencional (`feat`, `fix`, `docs`…).
- Cortos: línea de asunto y a lo mucho unas viñetas.
- Sin líneas de atribución a Claude ni a IA.
- La identidad de este repo es `felipegerard@gmail.com` (configurada como local).
