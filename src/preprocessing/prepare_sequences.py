# Flatten, padding, extracción de features
"""
prepare_sequences.py
---------------------
Convierte un esqueleto normalizado (T,17,2) en una secuencia lista
para la red:
1) Flatten joints fram-por-frame → (T, 34)
2) Padding o truncation a longitud fija (seq_len)
"""

import numpy as np

def prepare_sequence(skel_norm, config):
    """
    Params:
        skel_norm: np.ndarray (T,17,2)
        config: dict from config.yaml
        
    Output:
        seq: numpy (seq_len, 34)
    """
    T, J, C = skel_norm.shape
    seq_len = config["data"]["seq_len"]

    # 1. Flatten joints → (T, 34)
    seq = skel_norm.reshape(T, J*C)

    # 2. Padding o truncation
    if T < seq_len:
        # padding al final con ceros
        pad = np.zeros((seq_len - T, seq.shape[1]), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    else:
        # truncation
        seq = seq[:seq_len]

    return seq.astype(np.float32)
