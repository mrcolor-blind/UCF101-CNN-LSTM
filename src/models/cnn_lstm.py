# Arquitectura principal CNN+LSTM
# src/models/cnn_lstm.py

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    Modelo híbrido CNN + LSTM para reconocimiento de acciones
    basado en secuencias de esqueletos 2D ya preprocesadas.

    Entrada esperada:
        x: (batch, seq_len, feature_dim)
           donde feature_dim = 34 (17 joints * 2 coords)

    Flujo:
        1. Permutación → (batch, feature_dim, seq_len)
        2. CNN temporal → extracción de patrones locales
        3. Permutación → (batch, seq_len_reducido, channels)
        4. LSTM → modelado de dependencias largas
        5. FC → clasificación final

    Este módulo lee hiperparámetros desde config.yaml:
    model:
      cnn_filters: 64
      cnn_kernel: 3
      lstm_units: 128
      dropout: 0.3
      num_classes: X
    """

    def __init__(self, config):
        super().__init__()

        model_cfg = config["model"]
        filters = model_cfg["cnn_filters"]
        kernel = model_cfg.get("cnn_kernel", 3)
        lstm_units = model_cfg["lstm_units"]
        dropout = model_cfg["dropout"]
        num_classes = model_cfg["num_classes"]

        feature_dim = config["data"]["feature_dim"]  # típicamente 34

        # ---- CNN temporal ----
        # Conv1D recibe (batch, channels, seq_len)
        # channels = feature_dim (porque cada feature es un canal)
        self.conv1 = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=filters,
            kernel_size=kernel,
            padding=kernel // 2
        )

        self.bn1 = nn.BatchNorm1d(filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)  # reduce seq_len a la mitad

        # ---- LSTM ----
        # Recibe secuencias como (batch, seq_len, channels)
        self.lstm = nn.LSTM(
            input_size=filters,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # ---- Clasificador final ----
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_units, num_classes)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, 34)
        """

        # ---- BLOQUE CNN ----
        # permutar para Conv1D: (batch, 34, seq_len)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # ahora seq_len reducido

        # ---- BLOQUE LSTM ----
        # permutar de regreso: (batch, seq_len_reducido, filters)
        x = x.permute(0, 2, 1)

        # LSTM regresa output y (h_n, c_n)
        # nos quedamos con h_n[-1] como embedding final
        output, (h_n, c_n) = self.lstm(x)
        h_final = h_n[-1]  # shape: (batch, lstm_units)

        # ---- CLASIFICACIÓN ----
        logits = self.fc(h_final)
        return logits
