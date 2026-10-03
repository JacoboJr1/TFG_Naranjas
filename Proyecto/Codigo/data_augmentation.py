# ==============================================================================
# CÓDIGO PARA REALIZAR ALGUNOS AUMENTOS DE DATOS (SOLAMENTE USADO EN EL PRIMER YOLO)
# PROYECTO: Estudio y aplicación de YOLO, redes convolucionales (CNN) y modelos 
#           de lenguaje (LLM) para la detección, clasificación y tratamiento de naranjas
# ==============================================================================

import os
import cv2
import albumentations as A
import random

# === CONFIGURACIÓN DE RUTAS Y PARÁMETROS OPERATIVOS ===
# Directorio de entrada con las muestras fotográficas de cítricos originales
input_folder = "C:/Users/IVIA/Downloads/Citrus/Pruebas CITRUS 03-05-2018/Data_augmentation"
# Directorio destino optimizado para el almacenamiento del set extendido expandido
output_folder = "C:/Users/IVIA/Downloads/Data augmentation imagenes"
os.makedirs(output_folder, exist_ok=True)

# Factor multiplicador de expansión (Número de variantes sintéticas a generar por imagen)
num_augmented = 5

# DEFINICIÓN DEL PIPELINE DE TRANSFORMACIONES ESTOCÁSTICAS
# Configuración geométrica y pixelar para robustecer la invariancia del modelo CNN/YOLO
transform = A.Compose([
    A.RandomRotate90(),                 # Rotaciones ortogonales aleatorias (0°, 90°, 180°, 270°)
    A.HorizontalFlip(p=0.5),            # Inversión especular horizontal con probabilidad del 50%
    A.VerticalFlip(p=0.2),              # Inversión especular vertical con probabilidad del 20%
    A.RandomBrightnessContrast(p=0.5),  # Perturbación estocástica de las curvas de brillo y contraste
    A.Blur(blur_limit=3, p=0.2),        # Simulación de desenfoque por movimiento o pérdida de foco óptico
    A.LongestMaxSize(max_size=640),     # Redimensión proporcional basada en el eje de mayor longitud
    A.PadIfNeeded(min_height=640, min_width=640, border_mode=0)  # Relleno de bordes (Padding negro estático) hasta un formato cuadrado de 640x640
])

count_total = 0 # Contador analítico de imágenes sintéticas exportadas con éxito

# Procesamiento iterativo de los archivos gráficos en el directorio de entrada
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', 'bmp')):
        image_path = os.path.join(input_folder, filename)
        
        # Lectura de la matriz de la imagen en formato nativo BGR mediante OpenCV
        image = cv2.imread(image_path)

        # Control de excepciones frente a archivos corruptos o incompatibilidades de lectura
        if image is None:
            print(f"⚠️ Alerta: No se pudo leer o decodificar el archivo {filename}")
            continue

        # Transposición de espacios de color: conversión obligatoria de BGR a RGB para compatibilidad con Albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Subbucle iterativo para la generación masiva de variantes aumentadas
        for i in range(num_augmented):
            # Invocación de la tubería probabilística sobre el tensor de la imagen original
            augmented = transform(image=image)['image']
            
            # Control de estabilidad del proceso estocástico
            if augmented is None:
                print(f"❌ Falló la generación aumentada de {filename} (iteración {i})")
                continue

            # Reversión del espacio de color: reconversión de RGB a BGR para exportación estándar vía OpenCV
            augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            
            # Codificación sistemática del nuevo nombre de archivo indexado
            new_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
            out_path = os.path.join(output_folder, new_filename)

            # Volcado físico y compresión del nuevo subtensor gráfico en el almacenamiento local
            success = cv2.imwrite(out_path, augmented_bgr)
            if success:
                count_total += 1
            else:
                print(f"❌ Error de escritura: No se pudo almacenar {new_filename}")

print(f"✅ Data Augmentation finalizado. Se han incorporado con éxito {count_total} nuevas muestras al ecosistema.")