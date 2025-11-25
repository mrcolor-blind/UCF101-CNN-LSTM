# Manejo de rutas
"""
paths.py
---------
Funciones auxiliares para manejar rutas y cargar config.yaml
"""

import os
import yaml

def ensure_dir(path):
    """
    Crea un directorio si no existe.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(config_path="config/config.yaml"):
    """
    Carga archivo YAML de configuración.
    
    Returns:
        dict
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config
