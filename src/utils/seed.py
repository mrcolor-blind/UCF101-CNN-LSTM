# Seteo de random seeds
# src/utils/seed.py

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Fija seeds para reproducibilidad en Python, NumPy y PyTorch.
    Debe llamarse ANTES de crear el modelo, DataLoaders o entrenar.

    El determinismo en cuDNN puede hacer el entrenamiento más lento,
    pero garantiza reproducibilidad bit a bit.
    """

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Opciones de determinismo (puede ralentizar un poco)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[SEED] Seeds fijadas en {seed}")
