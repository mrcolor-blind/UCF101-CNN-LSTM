Autor: Miguel Angel Barrientos A01637150

# UCF101 – Action Recognition con Skeletons 2D  
## Implementación de un modelo Deep Learning (CNN + LSTM)

Este repositorio contiene una implementación completa para la clasificación de acciones humanas usando el dataset **UCF101 Skeleton 2D**, siguiendo la arquitectura **CNN + LSTM**, y un pipeline modular reproducible que cubre:
- Preprocesamiento del dataset
- Construcción del modelo
- Entrenamiento supervisado
- Evaluación
- Predicción individual
- Organización profesional del proyecto

### 2D Skeletons
https://mmaction2.readthedocs.io/en/latest/dataset_zoo/skeleton.html

### UCF
https://www.crcv.ucf.edu/data/UCF101.php

---

## 1. Objetivo del Proyecto
El objetivo es entrenar un modelo de deep learning capaz de reconocer acciones humanas en video utilizando **coordenadas 2D de esqueletos** (keypoints), extraídas del dataset UCF101 mediante modelos pose-estimation.  
Este enfoque reduce la dimensionalidad del video completo y permite entrenar modelos eficientes con recursos limitados.

El pipeline implementa:
- Extracción de keypoints (proporcionados por el dataset)
- Normalización y preprocesamiento
- Construcción de secuencias temporales
- Entrenamiento de redes híbridas CNN + LSTM

---

## 2. Estructura del Proyecto

```
ucf101-cnn-lstm/
│
├── main.py
├── README.md
├── requirements.txt
│
├── config/
│ ├── config.yaml
│ └── classes_subset.txt
│
├── data/
│ ├── raw/ → ucf101_2d.pkl
│ ├── processed/ → secuencias .npy preprocesadas
│ └── splits/ → train.csv / val.csv / test.csv
│
├── src/
│ ├── preprocessing/
│ │ ├── loader.py
│ │ ├── normalize.py
│ │ ├── prepare_sequences.py
│ │ └── build_dataset.py
│ │
│ ├── dataset/
│ │ └── skeleton_dataset.py
│ │
│ ├── models/
│ │ ├── cnn_lstm.py
│ │ └── baseline_lstm.py
│ │
│ ├── training/
│ │ ├── train.py
│ │ ├── evaluate.py
│ │ └── utils.py
│ │
│ ├── inference/
│ │ └── predict.py
│ │
│ └── utils/
│ ├── paths.py
│ ├── logger.py
│ └── seed.py
│
└── notebooks/
├── exploratory.ipynb
├── debug_sequences.ipynb
└── test_model.ipynb
```


---

## 3. Dataset: UCF101 Skeleton 2D

El dataset utilizado proviene de la versión esquelética de UCF101, donde a cada video se le ha extraído una representación basada en **keypoints 2D** usando un modelo de pose estimation.

Cada entrada del dataset contiene:
- `frame_dir` → nombre del video  
- `label` → clase de acción (entero)
- `keypoint` → matriz de forma **(M × T × 17 × 2)**  
  - M = número de personas  
  - T = número de frames  
  - 17 = keypoints estilo COCO  
  - 2 = coordenadas (x,y)
- `keypoint_score` → confianzas de detección (solo 2D)

El archivo maestro `ucf101_2d.pkl` contiene:
- un diccionario con `split`
- una lista de `annotations`

Cada `annotation` representa un video.

### 3.1 Clases

Aunque el dataset originalmente tiene 101 clases, se filtraron las siguientes 5 para tener una implementación más simple y eficiente:

 - ApplyEyeMakeup
 - JumpingJack
 - BasketballDunk
 - PushUps
 - YoYo


---
## 4. Preparación del repositorio

Antes de correr el pipeline, asegúrate de descargar las anotaciones del siguiente link:
https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl
y colocar `ucf101_2d.pkl` en `data/raw`.

Recuerda instalar los recursos necesarios:
```
pip install -r requirements.txt
```

### 4.1 Configuración importante (`config.yaml`)

El archivo `config.yaml` controla **todo el pipeline**, incluyendo rutas, selección de modelo, hiperparámetros, clases filtradas y parámetros del preprocesamiento.

A continuación se muestra un resumen de los campos más importantes:

#### **Selección del modelo**
Permite elegir fácilmente entre el modelo baseline y el modelo final:

```yaml
model:
  type: "cnn_lstm"        # opciones: "cnn_lstm" o "baseline_lstm"
  num_classes: 5
  lstm_units: 128
  dropout: 0.3
  cnn_filters: 64         # usado solo en cnn_lstm
  cnn_kernel: 3
```


## 5. Pipeline de Preprocesamiento

El preprocesamiento convierte cada anotación en una secuencia temporal lista para el modelo.

### Pasos principales:

1. Cargar `ucf101_2d.pkl`
2. Iterar sobre todas las anotaciones
3. Extraer keypoints principales (solo persona M=0)
4. Normalizar:
   - centrar en cadera/pelvis
   - escalar por altura por frame
   - smoothing temporal opcional
5. Convertir `(T, 17, 2)` → `(T, 34)`
6. Padding/truncation a `seq_len` fijo (configurable)
7. Guardar cada video como `.npy`
8. Generar splits estratificados:
   - `train.csv`
   - `val.csv`
   - `test.csv`

### Generación de nuevos splits
Aunque el dataset contiene varios splits (`train1`, `train2`, `train3`, `test1`, etc.), estos pertenecen a protocolos de evaluación académica y:
- no incluyen validation set  
- no son compatibles con el preprocesamiento (videos pueden fallar o cambiar longitud)  
- no permiten seleccionar solo algunas clases  
- no garantizan reproducibilidad con los `.npy` nuevos  

Por ello se generan splits **propios del pipeline**, completamente alineados con los datos preprocesados.

---

## 6. Modelos Implementados

### 6.1 Baseline: LSTM Simple

El modelo baseline consiste en una única capa **LSTM** seguida de una capa totalmente conectada.  
Su propósito es establecer un punto de referencia mínimo contra el cual comparar el desempeño del modelo final más avanzado.

**Características principales:**
- Entrada: secuencias `(seq_len, 34)`  
- Una sola capa `LSTM(hidden_size=128)`  
- `Dropout` para regularización  
- `Linear → Softmax` para clasificación  
- No captura patrones locales tan eficientemente como el modelo CNN+LSTM  
- Extremadamente eficiente y rápido de entrenar

**Objetivo del baseline:**  
Evaluar qué tan bien se comporta un modelo recurrente puro, sin convoluciones temporales, y cuantificar la mejora aportada por la arquitectura híbrida.


### 6.2 Modelo Final: CNN + LSTM (Híbrido Espacio–Temporal)

El modelo final utilizado para la clasificación de acciones humanas está basado en una arquitectura híbrida **CNN + LSTM**, diseñada específicamente para secuencias de esqueletos 2D provenientes del dataset UCF101.

#### Flujo del modelo

1. **Entrada**
   - Secuencias normalizadas de keypoints de forma `(seq_len, 34)`
   - Corresponde a 17 joints × 2 coordenadas (x, y)

2. **Bloque CNN Temporal**
   - Se aplica una `Conv1D` sobre la dimensión temporal
   - Extrae patrones de movimiento locales entre frames
   - Incluye `BatchNorm`, `ReLU` y `MaxPool1D` para mejorar estabilidad y reducir la longitud temporal

3. **Bloque LSTM**
   - Recibe la salida reducida de la CNN
   - Modela dependencias temporales de largo alcance
   - Genera una representación final del movimiento del sujeto

4. **Clasificador Final**
   - Dropout para regularización
   - Capa densa que proyecta la representación final hacia las clases seleccionadas (5 clases)
   - Softmax implícito en la pérdida y en la etapa de predicción

#### Ventajas del enfoque CNN + LSTM
- Aprovecha dependencias temporales cortas (**CNN**) y largas (**LSTM**)
- Computacionalmente eficiente comparado con modelos 3D-CNN
- Robusto al ruido en keypoints y a variaciones de duración
- Adecuado para datasets esqueléticos preprocesados

## 7. Ejecución del Pipeline

### 7.1 Preprocesamiento

```
python main.py preprocess
```

Genera:
- `.npy` preprocesados
- `dataset_index.csv`
- `train.csv`, `val.csv`, `test.csv`

---

### 7.2 Entrenamiento

```
python main.py train
```

Lee:
- modelo desde `config.yaml`
- dataset desde `data/processed/`
- splits desde `data/splits/`

Genera:
- checkpoints
- métricas
- logs de entrenamiento

---

### 7.3 Evaluación

```
python main.py evaluate
```

Produce:
- accuracy
- F1-score
- matriz de confusión
- reporte por clase

---

### 7.4 Predicción individual

```
python main.py predict --file ruta_al_video.npy
```

Muestra:
- probabilidad por clase
- clase final predicha

---
---

## 8. Configuración

El archivo `config/config.yaml` define:
- rutas del dataset
- número de clases
- longitud de secuencia
- hiperparámetros (lr, batch_size, epochs)
- arquitectura (cnn_lstm / baseline_lstm)

Ejemplo de parámetros típicos:
- seq_len: 100
- lstm_units: 128
- cnn_filters: 64
- learning_rate: 1e-3

---

## 9. Notebooks Incluidos

- **exploratory.ipynb**  
  Exploración del archivo `ucf101_2d.pkl`, shapes, ejemplos, visualización de keypoints.

---


