"""Entrena un modelo de embeddings a partir de un YAML de configuración.

    python scripts/run_train.py --config configs/mnist.yaml
    python scripts/run_train.py --config configs/synthetic.yaml --set backbone=resnet18 --set epochs=5
"""

import argparse

import yaml

from embeddings.config import Config
from embeddings.train import train


def parse_overrides(pairs: list[str]) -> dict:
    overrides = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        # yaml convierte "5" en int, "true" en bool y "[a, b]" en lista
        overrides[key] = yaml.safe_load(value)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", help="auto, cpu, mps o cuda; tiene prioridad sobre el YAML")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="CLAVE=VALOR",
        help="sobreescribe una clave del YAML; se puede repetir",
    )
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    if args.device:
        overrides["device"] = args.device
    train(Config.from_yaml(args.config, overrides))


# en Mac, con num_workers > 0, sin este guard el script se vuelve a lanzar a sí
# mismo en cada worker y se cuelga
if __name__ == "__main__":
    main()
