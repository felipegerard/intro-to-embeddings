"""Funciones de pérdida. ArcFace es la del curso; TripletLoss es punto extra."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """Softmax con margen angular (Deng et al., 2019).

    `W` tiene una fila por identidad de ENTRENAMIENTO: es el centro de esa
    identidad sobre la esfera. Solo sirve para generar gradiente; al terminar
    de entrenar se descarta, y con ella cualquier dependencia del número de
    clases.
    """

    def __init__(self, dim: int, n_classes: int, s: float = 30.0, m: float = 0.4):
        super().__init__()
        # cada fila es el centro de una clase en la esfera
        self.W = nn.Parameter(torch.randn(n_classes, dim))
        nn.init.xavier_normal_(self.W)
        self.s, self.m = s, m

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # z ya viene normalizado; normalizamos también W
        cos = F.linear(z, F.normalize(self.W, dim=1))
        # clamp: sin él, acos(±1) tiene gradiente infinito y la pérdida se
        # vuelve NaN a las pocas épocas
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        # el margen se suma SOLO en la clase correcta
        target = F.one_hot(labels, cos.size(1)).bool()
        logits = torch.where(target, torch.cos(theta + self.m), cos)
        return F.cross_entropy(self.s * logits, labels)


class TripletLoss(nn.Module):
    """Punto extra: triplet loss con minería batch-hard.

    Para cada ancla del batch se toma el positivo MÁS LEJANO de su misma
    identidad y el negativo MÁS CERCANO de otra identidad, y la pérdida es

        max(0, d(a, p) - d(a, n) + margin)

    promediada sobre las anclas.

    Ojo: batch-hard necesita que cada batch traiga varias imágenes por
    identidad. Con batches aleatorios casi ninguna ancla tiene positivo, así
    que también van a necesitar un sampler PK (P identidades × K imágenes).
    Tiene la misma firma que `ArcFace.forward`, así que en `train.py` se
    intercambia en una línea.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Punto extra: implementen la triplet loss con batch-hard.")
