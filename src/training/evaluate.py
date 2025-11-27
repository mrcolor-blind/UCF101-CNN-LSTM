# Métricas, matriz de confusión
# src/training/evaluate.py

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from src.dataset.skeleton_dataset import create_dataloader
from src.models.cnn_lstm import CNNLSTM
from src.models.baseline_lstm import BaselineLSTM
from src.utils.paths import load_config


def run_evaluation():
    """
    Evalúa el mejor modelo guardado (best_model.pt)
    usando el test set.
    Muestra:
        - Loss
        - Accuracy
        - Matriz de confusión
        - Classification report
    """

    print("\n[EVALUATE] Iniciando evaluación...\n")

    # ===========================
    # CONFIGURACIÓN
    # ===========================
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVALUATE] Usando dispositivo: {device}")

    # ===========================
    # DATA
    # ===========================
    test_loader = create_dataloader("test", config)

    # ===========================
    # MODELO
    # ===========================
    model_type = config["model"]["type"]
    if model_type == "cnn_lstm":
        model = CNNLSTM(config)
    elif model_type == "baseline_lstm":
        model = BaselineLSTM(config)
    else:
        raise ValueError(f"Modelo no reconocido: {model_type}")

    model = model.to(device)

    # cargar checkpoint
    ckpt_path = os.path.join(config["training"]["ckpt_path"], "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No se encontró el archivo de checkpoint: {ckpt_path}"
        )

    print(f"[EVALUATE] Cargando pesos desde: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # modo evaluación
    model.eval()

    # ===========================
    # EVALUACIÓN
    # ===========================
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq_batch, label_batch in tqdm(test_loader, desc="Evaluando"):
            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device)

            logits = model(seq_batch)
            loss = criterion(logits, label_batch)

            total_loss += loss.item() * seq_batch.size(0)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == label_batch).sum().item()
            total_samples += label_batch.size(0)

            # guardar para métricas
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())

    # ===========================
    # MÉTRICAS
    # ===========================
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print("\n=========== RESULTADOS ===========")
    print(f"Loss en test:     {avg_loss:.4f}")
    print(f"Accuracy en test: {accuracy:.4f}")

    # matriz de confusión
    print("\nMatriz de confusión:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    # clasificación por clase
    print("\nReporte por clase:")
    print(classification_report(all_labels, all_preds))

    print("\n[EVALUATE] Evaluación finalizada.\n")
