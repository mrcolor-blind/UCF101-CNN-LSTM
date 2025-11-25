# Normalización de coordenadas
"""
normalize.py
-------------
Normaliza un esqueleto:
1) Centrado en pelvis / joint base
2) Escalado por altura (normalización a escala unitaria)
3) Opcional: smoothing temporal
"""

import numpy as np
import scipy.ndimage

def normalize_skeleton(skel, smoothing=True):
    """
    Normaliza un esqueleto frame por frame.

    Params:
        skel: np.ndarray shape (T, 17, 2)
    """
    T, J, C = skel.shape

    # 1. Elegir joint base (puedes usar pelvis = joint 8, depende del dataset)
    base_joint = 8  
    base = skel[:, base_joint:base_joint+1, :]  # shape (T,1,2)

    # 2. Centrar coordenadas
    centered = skel - base  # broadcasting → resta a cada joint

    # 3. Escalar por altura del esqueleto
    # calcular distancia hombro–cadera o max_range en el frame
    height = np.max(centered[...,1] - np.min(centered[...,1]), axis=1)  # forma rápida
    height[height == 0] = 1  # evitar división entre 0

    height = height.reshape(T, 1, 1)
    normalized = centered / height

    # 4. Smoothing temporal (opcional)
    if smoothing:
        normalized = scipy.ndimage.gaussian_filter1d(
            normalized, sigma=1, axis=0
        )

    return normalized.astype(np.float32)
