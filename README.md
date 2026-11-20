# Clasificación de Piedra, Papel o Tijera con una CNN desde cero

Proyecto final de **Introducción al Aprendizaje Máquina** — Ingeniería en Inteligencia Artificial, FIME, UANL.

Red Neuronal Convolucional (CNN) diseñada y entrenada **desde cero** con TensorFlow/Keras, sin modelos preentrenados, para reconocer gestos de mano de piedra, papel o tijera a partir de imágenes.

## Integrantes

| Nombre | Matrícula |
|---|---|
| Luis Ricardo Salinas de la Peña | 2142924 |
| Alessandro Galvan Junquera | 2143543 |
| Peyton Oved Sanchez Olguin | 2152791 |
| Francisco Gabriel Delgado Ortiz | 2091878 |

## Dataset

[Rock-Paper-Scissors Images](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) (Kaggle, `drgfreeman/rockpaperscissors`)

- 2,188 imágenes RGB de 300×200 px sobre fondo verde
- 3 clases: `rock` (726), `paper` (712), `scissors` (750)

La notebook lee el dataset desde el `.zip` descargado de Kaggle (o lo descarga con [`kagglehub`](https://github.com/Kaggle/kagglehub) si no lo encuentra), así que no hace falta subir las imágenes al repositorio.

## Metodología

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

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy en prueba | 0.9818 |
| Loss en prueba | 0.0526 |
| Macro F1 en prueba | 0.9815 |

<!-- Después de ejecutar la notebook, sube la carpeta figuras/ y descomenta estas líneas:
![Curvas de aprendizaje](figuras/curvas_aprendizaje.png)
![Matriz de confusión](figuras/matriz_confusion.png)
-->

## Cómo ejecutarlo

### Google Colab

1. Descarga el dataset de [Kaggle](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) como `.zip`.
2. Haz clic en el botón **Open in Colab** de arriba.
3. Activa la GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)*.
4. Abre el panel de archivos (📁) y arrastra el `.zip`. Si tiene otro nombre que `rockpaperscissors.zip`, cámbialo en la variable `ZIP_PATH`.
5. Ejecuta todo: *Entorno de ejecución → Ejecutar todas*.

Si no subes el `.zip`, la notebook intenta descargar el dataset con `kagglehub`.




