# Modelo baseline LSTM simple
# src/models/baseline_lstm.py

import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    """
    Modelo baseline simple basado únicamente en LSTM.
    Diseñado para evaluar el desempeño mínimo esperado
    usando únicamente modelado temporal, sin convoluciones.

    Entrada esperada:
        x: (batch, seq_len, feature_dim)
           donde feature_dim = 34 (17 joints * 2 coords)

    Configuración vía config.yaml:
    model:
      lstm_units: 128
      dropout: 0.3
      num_classes: 5
    """

    def __init__(self, config):
        super().__init__()

        model_cfg = config["model"]
        lstm_units = model_cfg["lstm_units"]
        dropout = model_cfg["dropout"]
        num_classes = model_cfg["num_classes"]

        feature_dim = config["data"]["feature_dim"]  # típicamente 34

        # ---- LSTM ----
        self.lstm = nn.LSTM(
            input_size=feature_dim,
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
        x: (batch, seq_len, feature_dim)
        """
        output, (h_n, c_n) = self.lstm(x)
        h_final = h_n[-1]       # (batch, lstm_units)

        logits = self.fc(h_final)
        return logits
