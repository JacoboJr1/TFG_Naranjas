# ==============================================================================
# # CÓDIGO PARA ENTRENAR NUESTRO MODELO DE KERAS
# PROYECTO: Estudio y aplicación de YOLO, redes convolucionales (CNN) y modelos 
#           de lenguaje (LLM) para la detección, clasificación y tratamiento de naranjas
# ==============================================================================

import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import image_dataset_from_directory
from keras.preprocessing.image import load_img, img_to_array
import shutil

# === CONFIGURACIÓN DE HIPERPARÁMETROS Y ENTORNOS DE TRABAJO ===
IMG_SIZE = 512            # Resolución dimensional (ancho y alto) para la normalización de imágenes
BATCH_SIZE = 16           # Tamaño del lote para la propagación hacia adelante y atrás
EPOCHS = 200              # Límite máximo de iteraciones globales sobre el dataset
PATIENCE = 70             # Tolerancia del Early Stopping ante el estancamiento de la función de pérdida
DATASET_DIR = 'C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/crops_malas_para_clasificar_YOLO'  # Directorio con subcarpetas estructuradas por clase
MODEL_NAME = 'C:/Users/IVIA/Desktop/modelo_cnn_enfermedades_512.h5'                     # Ruta de exportación para la arquitectura entrenada
CARPETA_PREDICCION = 'C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/predicciones/predict_malas_128_enfermedades_2'  # Muestras desconocidas a categorizar

def entrenar_cnn():
    """
    Construye, compila y entrena un modelo Convolucional Secuencial.
    Gestiona el pipeline de datos estructurados e implementa optimización por minilotes.
    """
    # Instanciación y vectorización estructurada del subconjunto de entrenamiento
    train_ds_raw = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,       # Reserva un 20% del conjunto de datos para evaluación síncrona
        subset="training",
        seed=42,                     # Semilla pseudoaleatoria para asegurar la reproducibilidad
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    # Instanciación homóloga para el subconjunto de validación intermedia
    val_ds_raw = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    
    # Extracción analítica de las etiquetas de clase deducidas a partir de los subdirectorios
    class_names = train_ds_raw.class_names
    
    # Pipeline de normalización de píxeles en caliente (Mapeo a rango flotante de [0.0, 1.0])
    normalization_layer = lambda ds: ds.map(lambda x, y: (x / 255.0, y))
    train_ds = normalization_layer(train_ds_raw)
    val_ds = normalization_layer(val_ds_raw)
    
    # ARQUITECTURA DE LA RED NEURONAL CONVOLUCIONAL (CNN)
    model = Sequential([
        # Primer bloque convolucional: extracción de descriptores espaciales primitivos y bordes
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2), # Reducción bidimensional del mapa de características
        
        # Segundo bloque convolucional: abstracción intermedia de formas geométricas y contornos
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        # Tercer bloque convolucional: captura de texturas fitopatológicas complejas
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Flatten(), # Aplanamiento del tensor tridimensional a un vector unidimensional
        
        # Capas densas (Totalmente Conectadas) para el mapeo conceptual de alto nivel
        Dense(128, activation='relu'),
        Dropout(0.5), # Regularización estocástica por desactivación de nodos para mitigar el overfitting
        
        # Capa de salida lineal acoplada con Softmax para obtener distribuciones probabilísticas categóricas
        Dense(len(class_names), activation='softmax')
    ])
    
    # Compilación del modelo definiendo el optimizador adaptativo y la pérdida multiclase entera
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Callbacks de optimización: Early Stopping para detener el entrenamiento al alcanzar la convergencia
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    
    # Lanzamiento del algoritmo de optimización iterativa sobre el dataset normalizado
    model.fit(
        train_ds,
        # validation_data=val_ds, # Habilitar para activar la evaluación cruzada por época
        epochs=EPOCHS,
        # callbacks=[early_stop]   # Habilitar para prevenir el sobreajuste mediante parada temprana
    )
    
    # Exportación física del estado y pesos sinápticos óptimos del modelo en formato HDF5
    model.save(MODEL_NAME)
    print(f"✅ Arquitectura guardada con éxito en: {MODEL_NAME}")
    
    return model, class_names, val_ds

def predecir_enfermedades_validacion(val_ds, model, class_names):
    """
    Evalúa de forma determinista el rendimiento del modelo entrenado frente al set de validación,
    confrontando la etiqueta predictiva con la realidad empírica (ground truth).
    """
    print("\n--- Evaluando Dataset de Validación ---")
    
    total_aciertos = 0
    total_imagenes = 0

    # Descompresión secuencial de minilotes (tensores de imagen y vectores de etiquetas reales)
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0) # Inferencia masiva por lote sin salida redundante
        
        for i in range(len(images)):
            pred_idx = np.argmax(preds[i])       # Indice de la clase con máxima probabilidad asignada
            true_idx = labels[i].numpy()         # Transformación del tensor escalar a tipo nativo CPU
            
            clase_predicha = class_names[pred_idx]
            clase_real = class_names[int(true_idx)]
            confidence = np.max(preds[i])        # Nivel cuantitativo de confianza de la inferencia

            # Validación de coincidencia diagnóstica
            if pred_idx == true_idx:
                resultado = "✅"
                total_aciertos += 1
            else:
                resultado = "❌"

            print(f"{resultado} Real: {clase_real} | Predicho: {clase_predicha} ({confidence:.2f})")
            total_imagenes += 1

    # Cálculo métrico final de la precisión de validación global
    accuracy_final = (total_aciertos / total_imagenes) * 100
    print(f"\n📊 Resultado Final Validación: {total_aciertos}/{total_imagenes} aciertos ({accuracy_final:.2f}%)")

def predecir_enfermedades(carpeta_crops, model, class_names):
    """
    Pipeline de producción cualitativo: realiza inferencias sobre muestras externas no etiquetadas,
    y clasifica físicamente los archivos organizándolos en directorios especializados.
    """
    for img_name in os.listdir(carpeta_crops):
        if img_name.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(carpeta_crops, img_name)
            
            # Carga y adecuación dimensional de la muestra externa
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0               # Conversión matricial y reescalado flotante
            img_array = np.expand_dims(img_array, axis=0)       # Inserción de la dimensión del batch (1, H, W, C)

            # Ejecución del forward pass para inferir la patología
            pred = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(pred)
            confidence = np.max(pred)

            clase = class_names[pred_idx]
            print(f"{img_name}: {clase} ({confidence:.2f})")

            # Estructuración jerárquica del sistema de archivos según el diagnóstico
            clase_folder = os.path.join(carpeta_crops, clase)
            os.makedirs(clase_folder, exist_ok=True)

            # Reubicación y ordenación física de la muestra hacia el directorio de la patología detectada
            nueva_ruta = os.path.join(clase_folder, img_name)
            shutil.move(img_path, nueva_ruta)


if __name__ == "__main__":
    # Interruptor lógico operativo: True para reentrenar la CNN; False para cargar pesos existentes
    entrenar = True 

    if entrenar:
        model, class_names, val_ds = entrenar_cnn()
    else:
        model = load_model(MODEL_NAME)
        # Reconstrucción del mapa categórico analizando la distribución física de carpetas originales
        class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
        # Nota metodológica: Para ejecutar validación independiente en diferido, val_ds debe re-instanciarse aquí.
        _, val_ds = ... 

    # 1. Segmentación y ordenación de muestras desconocidas en base a inferencias estructurales
    predecir_enfermedades(CARPETA_PREDICCION, model, class_names)
    
    # 2. Análisis estadístico riguroso del comportamiento del clasificador frente al subconjunto de validación
    predecir_enfermedades_validacion(val_ds, model, class_names)