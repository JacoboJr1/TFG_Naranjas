# ==============================================================================
# PROYECTO: Estudio y aplicación de YOLO, redes convolucionales (CNN) y modelos 
#           de lenguaje (LLM) para la detección, clasificación y tratamiento de naranjas
# SOLO PREDECIR
# ==============================================================================

import os
import warnings
from ultralytics import YOLO
from multiprocessing import freeze_support

# Ocultar alertas de inicialización de librerías secundarias para limpiar la salida en consola
warnings.filterwarnings('ignore')

def predecir_imagenes(peso_modelo: str, carpeta_imagenes: str, carpeta_salida: str = "C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/predicciones/predict/predict2"):
    """
    Carga una red entrenada YOLOv11 para ejecutar el pipeline de inferencia sobre un directorio.
    Exporta de manera síncrona los resultados visuales (imágenes anotadas) y los descriptores geométricos (.txt).
    """
    # Garantizar la existencia física del directorio raíz de destino antes del volcado de datos
    os.makedirs(carpeta_salida, exist_ok=True)

    # Carga del modelo en memoria utilizando los pesos sinápticos óptimos (*.pt) del entrenamiento
    model = YOLO(peso_modelo)

    # Filtrado dinámico y compresión de rutas absolutas de imágenes válidas (formatos JPG y PNG)
    image_paths = [
        os.path.join(carpeta_imagenes, img)
        for img in os.listdir(carpeta_imagenes)
        if img.lower().endswith(('.jpg', '.png'))
    ]

    # Iteración secuencial sobre cada muestra gráfica del conjunto seleccionado para la inferencia
    for image_path in image_paths:
        results = model.predict(
            source=image_path,
            save=True,              # Exporta la imagen resultante con la bounding box y el identificador de clase dibujados
            save_txt=True,          # Genera un fichero .txt homólogo con las coordenadas relativas normalizadas [class, x, y, w, h]
            conf=0.5,               # Umbral de corte de confianza (Confidence Score Threshold) para filtrar detecciones débiles
            iou=0.7,                # Límite de Intersection over Union para el algoritmo de supresión de no máximos (NMS)
            project=carpeta_salida, # Especificación del directorio base para el almacenamiento de los resultados
            name="predict",         # Subcarpeta jerárquica destinada a albergar las salidas de la sesión actual
            exist_ok=True           # Permite la reutilización/escritura en el directorio sin lanzar excepciones del sistema
        )
        
        # Monitorización analítica por consola: despliega los tensores lógicos y numéricos de cada bounding box detectada
        print(f"Resultados de inferencia para {image_path}:")
        for r in results:
            print(r.boxes)

def main():
    """
    Orquestador de ejecución principal. Configura los entornos de entrada/salida y lanza el proceso.
    """
    # Definición de rutas absolutas del sistema para pesos y datos de validación
    ruta_pesos = "C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/runs/detect/train_prueba15_con_1800_img/weights/best.pt"  
    carpeta_imagenes = "C:/Users/IVIA/Desktop/TFG Jacobo/Datasets/TFG_Naranjas.v9-2800-buenas-2800-malas.yolov11/valid/images"
    
    # Invocación síncrona del pipeline de predicción masiva
    predecir_imagenes(ruta_pesos, carpeta_imagenes)

if __name__ == "__main__":
    # Asegura la estabilidad operativa y la correcta serialización de procesos en entornos concurrentes de Windows
    freeze_support()
    main()