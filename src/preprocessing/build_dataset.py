"""
build_dataset.py
-----------------
Construye el dataset final para entrenamiento, validación y prueba a partir
de las anotaciones 2D del archivo maestro ucf101_2d.pkl.

Este archivo ahora:
1) Carga el archivo maestro ucf101_2d.pkl
2) Itera sobre cada anotación (cada video)
3) Obtiene la secuencia (T,17,2)
4) Normaliza coordenadas
5) Aplica flatten y padding → (seq_len, 34)
6) Asigna etiqueta
7) Guarda los arrays en data/processed/
8) Genera splits train/val/test
"""

import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.normalize import normalize_skeleton
from src.preprocessing.prepare_sequences import prepare_sequence
from src.utils.paths import load_config, ensure_dir


# ============================================
# FUNCIÓN PRINCIPAL DEL PIPELINE
# ============================================

def run_preprocessing():
    """
    Orquesta todo el pipeline:
    - Carga config.yaml
    - Abre ucf101_2d.pkl
    - Procesa cada anotación como un video
    - Guarda dataset procesado
    - Crea splits train/val/test
    """
    print("\n[BUILD DATASET] Iniciando preprocesamiento...\n")

    config = load_config()

    raw_pkl_path = config["data"]["raw_pkl"]  # debe apuntar a data/raw/ucf101_2d.pkl
    processed_path = config["data"]["processed_path"]
    splits_path = "data/splits/"

    ensure_dir(processed_path)
    ensure_dir(splits_path)

    # 1. Cargar el archivo maestro
    print(f"Cargando anotaciones desde {raw_pkl_path} ...")
    with open(raw_pkl_path, "rb") as f:
        data = pickle.load(f)

    annotations = data["annotations"]
    print(f"Total de videos en anotaciones: {len(annotations)}\n")

    data_list = []

    # 2. Procesar cada anotación como si fuera un archivo individual
    for ann in tqdm(annotations, desc="Procesando videos"):
        try:
            sample = process_single_annotation(ann, config, processed_path)
            if sample is not None:
                data_list.append(sample)
        except Exception as e:
            print(f"ERROR procesando {ann.get('frame_dir','?')}: {e}")
            continue

    # 3. Guardar dataset index
    df = pd.DataFrame(data_list)
    index_csv = os.path.join(processed_path, "dataset_index.csv")
    df.to_csv(index_csv, index=False)
    print(f"\nIndex guardado en: {index_csv}\n")

    # 4. Crear splits
    create_splits(df, splits_path)

    print("\n[BUILD DATASET] Preprocesamiento finalizado\n")


# ============================================
# PROCESAR UNA ANOTACIÓN DEL PKL
# ============================================

def process_single_annotation(ann, config, processed_path):
    """
    Procesa una única anotación del pkl maestro.

    ann contiene:
        frame_dir (str) → id del video
        keypoint → (M,T,V,C)
        label (int)
        etc.

    Returns dict para dataset_index.csv
    """

    video_id = ann["frame_dir"]          # string del nombre del video
    raw_kp = ann["keypoint"]             # (M, T, 17, 2)
    label = ann["label"]                 # int

    # 1. Tomar solo la persona principal
    if raw_kp.ndim != 4:
        print(f"[WARNING] Video {video_id} tiene shape inválida: {raw_kp.shape}")
        return None

    # persona 0 → (T,17,2)
    skel = raw_kp[0].astype(np.float32)

    # 2. Normalizar
    skel_norm = normalize_skeleton(skel)

    # 3. Convertir a secuencia fija
    seq = prepare_sequence(skel_norm, config)

    # 4. Guardar .npy
    out_file = os.path.join(processed_path, f"{video_id}.npy")
    np.save(out_file, seq)

    return {
        "video_id": video_id,
        "path": out_file,
        "label": label
    }


# ============================================
# SPLITS TRAIN / VAL / TEST
# ============================================

def create_splits(df, splits_path, train_ratio=0.7, val_ratio=0.15):
    """
    Crea archivos CSV estratificados.
    """

    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df["label"],
        random_state=42
    )

    val_relative = val_ratio / (1 - train_ratio)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_relative),
        stratify=temp_df["label"],
        random_state=42
    )

    ensure_dir(splits_path)
    train_df.to_csv(os.path.join(splits_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_path, "val.csv"), index=False)
    test_df.to_csv(os.path.join(splits_path, "test.csv"), index=False)

    print("\nSplits creados:")
    print("  train.csv:", len(train_df))
    print("  val.csv:  ", len(val_df))
    print("  test.csv: ", len(test_df))
