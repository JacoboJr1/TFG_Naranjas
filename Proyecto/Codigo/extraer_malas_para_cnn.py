# ==============================================================================
# CÓDIGO PARA EXTRAER LAS NARANJAS MALAS PARA KERAS
# PROYECTO: Estudio y aplicación de YOLO, redes convolucionales (CNN) y modelos 
#           de lenguaje (LLM) para la detección, clasificación y tratamiento de naranjas
# ==============================================================================

import os
import cv2

# === CONFIGURACIÓN DE ENTORNOS Y RUTAS ABSOLUTAS ===
# Directorio raíz que aloja las imágenes originales del dataset
carpeta_predicciones = 'C:/Users/IVIA/Desktop/TFG Jacobo/Datasets/TFG_Naranjas.v9-2800-buenas-2800-malas.yolov11/train/images'  
# Directorio homólogo que contiene las etiquetas e inferencias en formato estructurado .txt
carpeta_labels = 'C:/Users/IVIA/Desktop/TFG Jacobo/Datasets/TFG_Naranjas.v9-2800-buenas-2800-malas.yolov11/train/labels'  
# Indexación numérica que identifica unívocamente la clase "mal estado" en la configuración YOLO
clase_mala_id = 1  
# Directorio destino optimizado para almacenar los subtensores correspondientes a los frutos afectados
carpeta_salida = 'C:/Users/IVIA/Desktop/TFG Jacobo/Pruebas/predicciones/predict_malas_512_enfermedades'  


# Garantizar la existencia física del directorio de destino antes del volcado de imágenes
os.makedirs(carpeta_salida, exist_ok=True)

# Procesamiento iterativo de los descriptores de anotación (.txt) en el subconjunto de etiquetas
for archivo in os.listdir(carpeta_labels):
    if archivo.endswith(".txt"):
        # Descomposición del nombre del archivo para emparejar el fichero .txt con su matriz .jpg correspondiente
        nombre_base = os.path.splitext(archivo)[0]
        ruta_imagen = os.path.join(carpeta_predicciones, nombre_base + ".jpg")
        ruta_txt = os.path.join(carpeta_labels, archivo)

        # Control de excepciones: omitir la iteración si el descriptor carece de su matriz gráfica asociada
        if not os.path.exists(ruta_imagen):
            continue  

        # Lectura y decodificación de la imagen digital mediante OpenCV
        imagen = cv2.imread(ruta_imagen)
        altura, ancho, _ = imagen.shape  # Extracción de las dimensiones de resolución nativa de la muestra

        # Apertura y lectura secuencial de las líneas del archivo de anotaciones YOLO
        with open(ruta_txt, "r") as f:
            for i, linea in enumerate(f):
                partes = linea.strip().split()
                # Validación de la integridad estructural del vector de datos (Clase, X, Y, W, H)
                if len(partes) < 5:
                    continue

                # Filtrado lógico: discriminar y omitir instancias pertenecientes a clases no patológicas
                clase_id = int(partes[0])
                if clase_id != clase_mala_id:
                    continue  

                # Desnormalización matemática de las coordenadas relativas proporcionadas por YOLO
                # Convierte los centroides y dimensiones porcentuales en píxeles absolutos
                x_center = float(partes[1]) * ancho
                y_center = float(partes[2]) * altura
                w = float(partes[3]) * ancho
                h = float(partes[4]) * altura

                # Cálculo de vértices geométricos espaciales (Coordenadas extremas de la caja de delimitación)
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                # Tratamiento de contorno y normalización de bordes: asegura que el recorte no desborde la imagen
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(ancho, x2), min(altura, y2)

                # Segmentación matricial (Slicing) de la región de interés (ROI) y guardado en disco
                crop = imagen[y1:y2, x1:x2]
                nombre_crop = f"{nombre_base}_mala_{i}.jpg"
                cv2.imwrite(os.path.join(carpeta_salida, nombre_crop), crop)

print(f"✅ Extracción completada. Subtensores de frutos patológicos almacenados con éxito en: {carpeta_salida}")