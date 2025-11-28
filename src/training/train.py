# Epoch loop, loss, optim, scheduler
# src/training/train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from src.dataset.skeleton_dataset import create_dataloader
from src.models.cnn_lstm import CNNLSTM
from src.models.baseline_lstm import BaselineLSTM
from src.utils.paths import load_config, ensure_dir
from src.utils.logger import logger


def run_training():
    """
    Función principal de entrenamiento.
    - Carga configuración
    - Instancia DataLoaders
    - Crea modelo según config["model"]["type"]
    - Corre loop de entrenamiento
    - Guarda checkpoints de mejor modelo
    """

    config = load_config()

    # ===========================
    #  CONFIGURAR LOGGER
    # ===========================
    log_path = config["logging"]["log_file"]
    logger.log_file = log_path
    logger.info("Iniciando entrenamiento...")

    # ===========================
    #  CONFIGURACIÓN GENERAL
    # ===========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")

    # ===========================
    # DATA
    # ===========================
    train_loader = create_dataloader("train", config)
    val_loader = create_dataloader("val", config)
    logger.info("DataLoaders cargados correctamente.")

    # ===========================
    # MODELO
    # ===========================
    model_type = config["model"]["type"]
    logger.info(f"Inicializando modelo: {model_type}")

    if model_type == "cnn_lstm":
        model = CNNLSTM(config)
    elif model_type == "baseline_lstm":
        model = BaselineLSTM(config)
    else:
        logger.error(f"Modelo desconocido: {model_type}")
        raise ValueError(f"Modelo no reconocido: {model_type}")

    model = model.to(device)

    # ===========================
    # ENTRENAMIENTO
    # ===========================
    epochs = config["training"]["epochs"]
    lr = config["training"]["lr"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        factor=0.5
    )

    ckpt_dir = config["training"]["ckpt_path"]
    ensure_dir(ckpt_dir)
    best_val_loss = float("inf")
    best_model_path = os.path.join(ckpt_dir, f"best_{model_type}.pt")

    logger.info(f"Entrenamiento por {epochs} épocas.")

    # ===========================
    # LOOP DE ÉPOCAS
    # ===========================
    for epoch in range(1, epochs + 1):

        logger.info(f"\n====== Época {epoch}/{epochs} ======")

        # -----------------------
        # TRAIN
        # -----------------------
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for seq_batch, label_batch in tqdm(train_loader, desc=f"Entrenando {epoch}/{epochs}"):

            seq_batch = seq_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()

            logits = model(seq_batch)
            loss = criterion(logits, label_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * seq_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label_batch).sum().item()
            total += label_batch.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total

        logger.info(f"[TRAIN] Loss={avg_train_loss:.4f} | Acc={train_acc:.4f}")

        # -----------------------
        # VALIDACIÓN
        # -----------------------
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for seq_batch, label_batch in tqdm(val_loader, desc="Validando"):
                seq_batch = seq_batch.to(device)
                label_batch = label_batch.to(device)

                logits = model(seq_batch)
                loss = criterion(logits, label_batch)

                val_loss += loss.item() * seq_batch.size(0)
                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == label_batch).sum().item()
                val_total += label_batch.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        logger.info(f"[VAL] Loss={avg_val_loss:.4f} | Acc={val_acc:.4f}")

        scheduler.step(avg_val_loss)

        # -----------------------
        # CHECKPOINT
        # -----------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.success(f"Nuevo mejor modelo guardado en {best_model_path}")

    logger.info("Entrenamiento finalizado.")
    logger.success(f"Mejor modelo con val_loss={best_val_loss:.4f} guardado en: {best_model_path}")
