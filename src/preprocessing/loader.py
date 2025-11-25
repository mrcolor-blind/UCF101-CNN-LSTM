# Cargar pkl → (T,17,2)

"""
loader.py
---------
Funciones para cargar archivos .pkl con esqueletos 2D.
"""

import pickle
import numpy as np

import pickle
import numpy as np

import pickle
import numpy as np

def load_skeleton_pkl(path, video_id):
    """
    Carga un video específico desde ucf101_2d.pkl
    
    Args:
        path: ruta al pkl maestro (ucf101_2d.pkl)
        video_id: nombre del video, ej: 'v_ApplyLipstick_g01_c01'
    
    Returns:
        skel: np.ndarray con shape (T, 17, 2)
    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        annotations = data["annotations"]

        # Buscar el video deseado
        for ann in annotations:
            if ann["frame_dir"] == video_id:
                kp = ann["keypoint"]      # (M, T, V, C)
                
                # Tomar persona principal (persona 0)
                kp = kp[0]                # (T, V, C)

                return kp.astype(np.float32)

        print(f"[LOADER] Video {video_id} no encontrado en {path}")
        return None

    except Exception as e:
        print(f"[LOADER ERROR] No se pudo cargar {path}: {e}")
        return None


def load_all_skeletons(path):
    """
    Carga TODAS las anotaciones del archivo ucf101_2d.pkl.

    ESTA FUNCIÓN NO SE USA, PUES EL PIPELINE CARGA CADA VIDEO INDIVIDUALMENTE, CON LA FUNCIÓN ANTERIOR
    
    Returns:
        dict: { video_id : (T,17,2), ... }
    """
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        annotations = data["annotations"]
        videos = {}

        for ann in annotations:
            video_id = ann["frame_dir"]
            kp = ann["keypoint"]      # (M, T, V, C)

            # Tomamos solo 1 persona (la principal)
            kp = kp[0]                # (T,17,2)

            videos[video_id] = kp.astype(np.float32)

        return videos

    except Exception as e:
        print(f"[LOADER ERROR] {e}")
        return {}
