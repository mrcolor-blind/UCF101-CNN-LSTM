# src/training/evaluate.py

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from datetime import datetime

from src.dataset.skeleton_dataset import create_dataloader
from src.models.cnn_lstm import CNNLSTM
from src.models.baseline_lstm import BaselineLSTM
from src.utils.paths import load_config, ensure_dir


def run_evaluation():
    """
    Evalúa el mejor modelo guardado usando el test set.
    Guarda:
        - Timestamp
        - Loss
        - Accuracy
        - Matriz de confusión
        - Classification report
    en:
        results/{model_type}_evaluation.txt
    """

    print("\n[EVALUATE] Iniciando evaluación...\n")

    # ===========================
    # CONFIGURACIÓN
    # ===========================
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVALUATE] Usando dispositivo: {device}")

    # carpeta results/
    results_dir = "results/"
    ensure_dir(results_dir)

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

    ckpt_path = os.path.join(config["training"]["ckpt_path"], f"best_{model_type}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No se encontró checkpoint: {ckpt_path}")

    print(f"[EVALUATE] Cargando pesos desde: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
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

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_batch.cpu().numpy())

    # ===========================
    # MÉTRICAS
    # ===========================
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    # ===========================
    # GUARDAR RESULTADOS
    # ===========================
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = os.path.join(results_dir, f"{model_type}_evaluation.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=========== RESULTADOS ===========\n\n")
        f.write(f"Fecha y hora de evaluación: {timestamp}\n\n")
        f.write(f"Loss en test:     {avg_loss:.4f}\n")
        f.write(f"Accuracy en test: {accuracy:.4f}\n\n")

        f.write("Matriz de confusión:\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Reporte por clase:\n")
        f.write(report)
        f.write("\n\n")

    print(f"\n[EVALUATE] Evaluación finalizada. Resultados guardados en:\n{output_path}\n")
