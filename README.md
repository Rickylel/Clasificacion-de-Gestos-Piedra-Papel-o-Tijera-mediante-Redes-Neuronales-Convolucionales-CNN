# ✊✋✌️ Clasificación de Piedra, Papel o Tijera con una CNN desde cero

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USUARIO/NOMBRE-REPO/blob/main/PIA_RPS_CNN.ipynb)

Proyecto final de **Introducción al Aprendizaje Máquina** — Ingeniería en Inteligencia Artificial, FIME, UANL.

Red Neuronal Convolucional (CNN) diseñada y entrenada **desde cero** con TensorFlow/Keras, sin modelos preentrenados, para reconocer gestos de mano de piedra, papel o tijera a partir de imágenes.

## 👥 Integrantes

| Nombre | Matrícula |
|---|---|
| Nombre Apellido | 0000000 |
| Nombre Apellido | 0000000 |
| Nombre Apellido | 0000000 |

## 📂 Dataset

[Rock-Paper-Scissors Images](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) (Kaggle, `drgfreeman/rockpaperscissors`)

- 2,188 imágenes RGB de 300×200 px sobre fondo verde
- 3 clases: `rock` (726), `paper` (712), `scissors` (750)

La notebook lee el dataset desde el `.zip` descargado de Kaggle (o lo descarga con [`kagglehub`](https://github.com/Kaggle/kagglehub) si no lo encuentra), así que no hace falta subir las imágenes al repositorio.

## 🧠 Metodología

1. **Preprocesamiento:** división estratificada 70 % / 15 % / 15 % (train / validation / test), redimensionamiento a 150×150 px y normalización de píxeles a [0, 1] (división entre 255).
2. **Arquitectura (desde cero):**

```
Input (150×150×3)
→ Aumento de datos: RandomFlip, RandomRotation(0.1), RandomZoom(0.1)
→ Conv2D(32)  + MaxPooling2D
→ Conv2D(64)  + MaxPooling2D
→ Conv2D(128) + MaxPooling2D
→ Conv2D(128) + MaxPooling2D
→ Flatten → Dropout(0.5) → Dense(256, ReLU) → Dense(3, Softmax)
```

3. **Entrenamiento:** Adam (lr = 0.001), `categorical_crossentropy`, métrica `accuracy`, batch size 32, 25 épocas.
4. **Evaluación:** curvas de pérdida y precisión, métricas en el conjunto de prueba, matriz de confusión, reporte de clasificación (Precision, Recall, F1 por clase) y ejemplos visuales de aciertos y errores.

## 📊 Resultados

| Métrica | Valor |
|---|---|
| Accuracy en prueba | _completar_ |
| Loss en prueba | _completar_ |
| Macro F1 en prueba | _completar_ |

<!-- Después de ejecutar la notebook, sube la carpeta figuras/ y descomenta estas líneas:
![Curvas de aprendizaje](figuras/curvas_aprendizaje.png)
![Matriz de confusión](figuras/matriz_confusion.png)
-->

## 🚀 Cómo ejecutarlo

### Opción 1: Google Colab (recomendada)

1. Descarga el dataset de [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) como `.zip`.
2. Haz clic en el botón **Open in Colab** de arriba.
3. Activa la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*.
4. Abre el panel de archivos (📁) y arrastra el `.zip`. Si tiene otro nombre que `rockpaperscissors.zip`, cámbialo en la variable `ZIP_PATH`.
5. Ejecuta todo: *Entorno de ejecución → Ejecutar todas*.

Si no subes el `.zip`, la notebook intenta descargar el dataset con `kagglehub`.

### Opción 2: local

```bash
git clone https://github.com/USUARIO/NOMBRE-REPO.git
cd NOMBRE-REPO
pip install -r requirements.txt
jupyter notebook PIA_RPS_CNN.ipynb
```

Coloca el `.zip` del dataset en la misma carpeta que la notebook (o ajusta `ZIP_PATH`).

## 📁 Estructura del repositorio

```
├── PIA_RPS_CNN.ipynb     # Notebook con la solución completa (secciones 1–4)
├── requirements.txt      # Dependencias para ejecución local
├── README.md
└── figuras/              # (se genera al ejecutar) gráficas y métricas para el reporte
```

## 📚 Referencias

- de la Bruère-Terreault, J. (2018). *Rock-Paper-Scissors Images* [Dataset]. Kaggle.
- Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE, 86*(11), 2278–2324.
