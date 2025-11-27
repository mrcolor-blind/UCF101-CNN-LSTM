# PyTorch Dataset / collate_fn
# src/dataset/skeleton_dataset.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SkeletonDataset(Dataset):
    """
    Dataset para cargar secuencias esqueléticas ya preprocesadas.

    Cada elemento contiene:
    - seq: tensor float32 con shape (seq_len, feature_dim)
    - label: tensor long con la clase

    Los datos provienen de:
        data/splits/{train,val,test}.csv
        data/processed/{video}.npy
    """

    def __init__(self, split_csv_path):
        super().__init__()
        self.df = pd.read_csv(split_csv_path)

        if len(self.df) == 0:
            raise ValueError(f"CSV vacío: {split_csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # cargar secuencia procesada
        seq_path = row["path"]
        seq_np = np.load(seq_path).astype(np.float32)
            
        # convertir a tensores
        seq_tensor = torch.from_numpy(seq_np)               # (seq_len, 34)
        label_tensor = torch.tensor(row["label"], dtype=torch.long)

        return seq_tensor, label_tensor


def get_split_df(split_name, config):
    """
    Regresa el dataframe del split deseado.
    split_name: "train", "val", "test"
    """
    splits_folder = config["data"]["splits_path"]
    csv_path = os.path.join(splits_folder, f"{split_name}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encuentra el split {csv_path}")

    return csv_path


def create_dataloader(split_name, config):
    """
    Crea un DataLoader a partir de un split.

    split_name: "train", "val", "test"
    """
    csv_path = get_split_df(split_name, config)

    dataset = SkeletonDataset(csv_path)

    batch_size = config["training"]["batch_size"]
    shuffle = True if split_name == "train" else False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )

    return loader
