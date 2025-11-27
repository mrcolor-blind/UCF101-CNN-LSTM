# Cargar un archivo y hacer predicción
# src/inference/predict.py

import os
import numpy as np
import torch
import torch.nn.functional as F

from src.models.cnn_lstm import CNNLSTM
from src.models.baseline_lstm import BaselineLSTM
from src.utils.paths import load_config


def load_class_names(config):
    """
    Lee las clases desde config/classes_subset.txt y regresa
    una lista donde:
        classes[i] = nombre de clase correspondiente al índice i
    """
    classes_file = config["data"]["classes_subset"]
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def run_prediction(input_file):
    """
    Predice la clase de una secuencia .npy ya preprocesada.
    """
    print("\n[PREDICT] Iniciando predicción...\n")

    # ===========================
    # CONFIG
    # ===========================
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PREDICT] Usando dispositivo: {device}")

    # ===========================
    # CARGAR CLASES
    # ===========================
    class_names = load_class_names(config)
    num_classes = len(class_names)

    # ===========================
    # CARGAR MODELO
    # ===========================
    model_type = config["model"]["type"]
    if model_type == "cnn_lstm":
        model = CNNLSTM(config)
    elif model_type == "baseline_lstm":
        model = BaselineLSTM(config)
    else:
        raise ValueError(f"Modelo no reconocido: {model_type}")

    model = model.to(device)

    ckpt_path = os.path.join(config["training"]["ckpt_path"], "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No existe el checkpoint: {ckpt_path}")

    print(f"[PREDICT] Cargando pesos desde {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ===========================
    # CARGAR ARCHIVO .npy
    # ===========================
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"No se encuentra el archivo: {input_file}")

    print(f"[PREDICT] Cargando secuencia desde: {input_file}")
    seq_np = np.load(input_file).astype(np.float32)  # (seq_len, 34)
    seq_tensor = torch.from_numpy(seq_np).unsqueeze(0).to(device)  # → (1, seq_len, 34)

    # ===========================
    # FORWARD PASS
    # ===========================
    with torch.no_grad():
        logits = model(seq_tensor)           # (1, num_classes)
        probs = F.softmax(logits, dim=1)     # probabilidades
        probs_np = probs.cpu().numpy()[0]

    pred_idx = int(np.argmax(probs_np))
    pred_class = class_names[pred_idx]

    # ===========================
    # OUTPUT
    # ===========================
    print("\n======= RESULTADO DE PREDICCIÓN =======")
    print(f"Clase predicha (índice): {pred_idx}")
    print(f"Clase predicha (nombre): {pred_class}")
    print("\nProbabilidades por clase:")
    for i, p in enumerate(probs_np):
        print(f"  {i} - {probs_np[i]}: {p:.4f}")

    print("\n[PREDICT] Predicción completada.\n")

