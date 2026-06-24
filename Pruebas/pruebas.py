# ==============================================================================
# CÓDIGO QUE SIRVE PARA ENTRENAR NUESTROS MODELOS YOLOv11
# Proyecto: Estudio y aplicación de YOLO, redes convolucionales (CNN) y modelos de lenguaje (LLM) para la detección, clasificación y tratamiento de naranjas
# ==============================================================================

import os
import warnings
import shutil
from ultralytics import YOLO
from multiprocessing import freeze_support

# Desactivar alertas innecesarias de librerías secundarias para limpiar la salida en consola
warnings.filterwarnings('ignore')

def entrenar_modelo(peso_modelo: str, yaml_path: str) -> YOLO:
    """
    Inicializa y ejecuta el proceso de entrenamiento de la arquitectura YOLOv11.
    Aplica hiperparámetros de optimización avanzados y aumentos de datos.
    """
    # Carga de los pesos preentrenados del modelo base (Transfer Learning)
    model = YOLO(peso_modelo)
    
    # Configuración del ciclo de entrenamiento (Ultralytics)
    model.train(
        data=yaml_path,       # Ruta al archivo de configuración de los datos (.yaml)
        epochs=200,          # Límite máximo de épocas de entrenamiento
        imgsz=600,           # Dimensionamiento dimensional de entrada para las imágenes
        optimizer='Adam',    # Optimizador matemático adaptativo para el ajuste de gradientes
        batch=-1,            # Autotuning: YOLO calcula el tamaño de batch máximo según la VRAM
        lr0=0.0005,          # Tasa de aprendizaje inicial (Learning Rate)
        lrf=0.01,            # Tasa de aprendizaje final (Fracción de lr0)
        weight_decay=0.0001, # Regularización L2 para mitigar el sobreajuste (overfitting)
        device=0,            # Ejecución forzada en la GPU principal indexada (CUDA)
        patience=70,         # Early Stopping: Detiene el entrenamiento si no hay mejora en 70 épocas
        cos_lr=True,         # Planificador de tasa de aprendizaje mediante descenso armónico/coseno
        amp=True,            # Mixed Precision (AMP): Acelera el cómputo reduciendo flotantes a FP16
        val=True,            # Ejecuta la validación de manera síncrona al finalizar cada época
        cache=True,          # Almacena el dataset en RAM/Caché para omitir cuellos de botella de disco
        verbose=True,        # Muestra en consola la métrica detallada por cada clase en cada ciclo
        augment=True,        # Activa las técnicas nativas de aumento de datos en caliente
        
        # Gestión y direccionamiento de directorios de salida
        project="C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/runs/detect",
        name="detect",
        exist_ok=True        # Sobrescribe o reutiliza la carpeta si ya existe sin lanzar error
    )
    return model

def validar_modelo(model: YOLO, carpeta_salida_val: str):
    """
    Evalúa el rendimiento intermedio del modelo utilizando el subconjunto de validación (Val Set)
    durante las fases de ajuste de hiperparámetros.
    """
    results = model.val(
        project=carpeta_salida_val, # Directorio base para el volcado de la validación
        name="validacion_oficial",  # Subcarpeta específica para este proceso
        exist_ok=True,
        save=True                   # Almacena en disco los gráficos estadísticos y muestras visuales
    )
    return results

def testear_modelo(model: YOLO, yaml_path: str, carpeta_salida_test: str):
    """
    Evaluación final y ciega del modelo utilizando exclusivamente el subconjunto de test (Test Set).
    Genera las métricas definitivas (mAP50, mAP50-95, Precisión, Recall) para la memoria del TFG.
    """
    print("\n--- Iniciando Evaluación Final Absoluta (Test Set) ---")
    results = model.val(
        data=yaml_path,
        split='test',             # Fuerza a la arquitectura a leer el vector 'test' del archivo YAML
        project=carpeta_salida_test,
        name="test_final",
        exist_ok=True,
        save=True                 # Exporta matrices de confusión finales, curvas P-R y F1-score
    )
    return results

def predecir_imagenes(model: YOLO, carpeta_imagenes: str, carpeta_salida_base: str):
    """
    Ejecuta el pipeline de inferencia sobre imágenes reales nunca antes vistas por el modelo.
    Genera el archivo visual con las cajas de delimitación y etiquetas predichas.
    """
    # Filtrado y compresión de rutas válidas en el directorio de entrada (Formatos JPG y PNG)
    image_paths = [
        os.path.join(carpeta_imagenes, img)
        for img in os.listdir(carpeta_imagenes)
        if img.lower().endswith(('.jpg', '.png'))
    ]

    # Bucle iterativo de inferencia por cada muestra
    for image_path in image_paths:
        results = model.predict(
            source=image_path,
            save=True,            # Exporta la imagen resultante con la caja de delimitación dibujada
            conf=0.5,             # Umbral mínimo de confianza (Confidence Score Threshold)
            iou=0.7,              # Umbral de Intersection over Union para la supresión de no máximos (NMS)
            project=carpeta_salida_base,
            name="predict",
            exist_ok=True
        )
        
        # Muestra por consola las coordenadas numéricas y tensores de las cajas predichas
        print(f"Resultados de inferencia para {image_path}:")
        for r in results:
            print(r.boxes)

def main():
    """
    Orquestador principal que define las variables de entorno, rutas absolutas
    y ejecuta de forma secuencial las fases del ciclo de vida del modelo de aprendizaje profundo.
    """
    # Definición de rutas absolutas del sistema (Entradas)
    ruta_pesos = "C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/versiones_YOLO/YOLO_2/yolo11n.pt"
    ruta_yaml = "C:/Users/IVIA/Desktop/TFG Jacobo/TFG_Naranjas_2.v3-800-imagenes-yolo-2_3.yolov11/data.yaml"
    
    # Estructura jerárquica de rutas de salida para resultados de ingeniería
    ruta_entrenamiento = "C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/runs/detect"
    ruta_validacion = "C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/runs/val"
    ruta_test = "C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/runs/test" 
    ruta_predicciones_base = "C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/predicciones/predict/predict_YOLO_2_automatico"

    # FASE 1: Entrenamiento y optimización de parámetros
    modelo = entrenar_modelo(ruta_pesos, ruta_yaml)
    
    # FASE 2: Validación intermedia (Control de Overfitting)
    resultados_val = validar_modelo(modelo, ruta_validacion)
    
    # FASE 3: Testeo estadístico riguroso (Métricas oficiales de rendimiento para el TFG)
    resultados_test = testear_modelo(modelo, ruta_yaml, ruta_test)
    
    # FASE 4: Inferencia y explotación cualitativa (Generación de muestras visuales)
    carpeta_test_imagenes = "C:/Users/IVIA/Desktop/TFG Jacobo/TFG_Naranjas_2.v3-800-imagenes-yolo-2_3.yolov11/test/images"
    predecir_imagenes(modelo, carpeta_test_imagenes, ruta_predicciones_base)

if __name__ == "__main__":
    # Soporte obligatorio para asegurar la correcta serialización de procesos en arquitecturas Windows
    freeze_support()
    main()