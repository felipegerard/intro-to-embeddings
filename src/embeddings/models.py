"""Backbones intercambiables y el head que produce el embedding."""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

BACKBONES = ("small_cnn", "resnet18", "mobilenet_v3_small")


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


def build_backbone(
    name: str,
    pretrained: bool = True,
    in_channels: int = 3,
) -> tuple[nn.Module, int]:
    """Devuelve `(backbone, dimensión de su vector de salida)`."""
    if name == "small_cnn":
        # desde cero, pensada para 28x28 o 32x32 en gris; corre bien en CPU
        net = nn.Sequential(
            _conv_block(in_channels, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            # promedia lo que quede del mapa espacial: acepta cualquier tamaño de imagen
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        return net, 128

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        feat_dim = net.fc.in_features
        # quitamos la capa que clasifica las 1000 clases de ImageNet: nos quedamos
        # con el vector que venía justo antes
        net.fc = nn.Identity()
        return net, feat_dim

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        feat_dim = net.classifier[0].in_features
        net.classifier = nn.Identity()
        return net, feat_dim

    raise ValueError(f"Backbone desconocido: {name!r}. Opciones: {', '.join(BACKBONES)}")


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 128,
        pretrained: bool = True,
        in_channels: int = 3,
    ):
        super().__init__()
        # los preentrenados en ImageNet siempre esperan 3 canales
        self.in_channels = in_channels if backbone == "small_cnn" else 3
        self.backbone, feat_dim = build_backbone(backbone, pretrained, self.in_channels)
        # head base: una capa oculta y salida a `embedding_dim`
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        # una imagen en gris que va a un backbone de 3 canales se copia en los 3
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.expand(-1, 3, -1, -1)
        z = self.head(self.backbone(x))
        # sin esta línea nada de lo demás funciona: el coseno, ArcFace y la
        # evaluación suponen vectores de norma 1
        return F.normalize(z, p=2, dim=1)
