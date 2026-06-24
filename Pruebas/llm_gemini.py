# ==============================================================================
# CÓDIGO DE USO DEL LLM GEMINI
# PROYECTO: Estudio y aplicación de YOLO, redes convolucionales (CNN) y modelos 
#           de lenguaje (LLM) para la detección, clasificación y tratamiento de naranjas
# ==============================================================================

import os
import google.generativeai as genai
from dotenv import load_dotenv

# === CONFIGURACIÓN DE CRIDENCIALES Y PROTOCOLOS DE SEGURIDAD ===
# Carga asíncrona de las variables de entorno locales (.env)
load_dotenv()
# Extracción de la clave de autenticación para los servicios de Google AI
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

# INSTANCIACIÓN DEL MODELO DE FRONTERA MULTIMODAL
# Modelo de arquitectura Gemini 2.5 Flash
model = genai.GenerativeModel("gemini-2.5-flash")

# Ruta absoluta del subtensor (crop) de la naranja patológica a evaluar por el LLM
image_path = "C:/Users/jacob/Desktop/TFG Jacobo/Pruebas/predicciones/predict_malas_128_enfermedades_2/otros_defectos/1_CAM1_33997929638_O_aug_0_jpg.rf.b333c75be388a7f049631d8f2c9d6c04_mala_0.jpg"

# 🔹 LECTURA DEL ARCHIVO DE LA IMAGEN (Añadido para que exista 'image_bytes')
with open(image_path, "rb") as img_file:
    image_bytes = img_file.read()

# INGENIERÍA DE PROMPTS (PROMPT ENGINEERING)
# Escribir el prompt que sea
prompt_base = """
Eres un experto en clasificación de defectos en naranjas.
Las categorías posibles son:
1. Manchas negras 
2. Hongo
3. Canker 
4. Scab 
5. Otros defectos
6. Trips
7. Greening
En este caso es una naranja donde presenta otros defectos.
Para cada imagen:
- Recomienda tratamiento (recuperar, procesar o descartar)
"""

# CONTROL DE EXCEPCIONES Y FLUJO DE INFERENCIA
try:
    # Ejecución de la consulta multimodal inyectando el tensor binario y las directrices textuales
    response = model.generate_content(
        [
            {"mime_type": "image/jpeg", "data": image_bytes}, # Estructuración de los metadatos de la imagen
            prompt_base                                      # Contexto e instrucciones lógicas
        ]
    )
    # Volcado en consola de la respuesta cualitativa generada por el modelo de lenguaje
    print(response.text)

except Exception as e:
    # Captura de errores de red, cuotas de API o problemas de codificación de archivos
    print("Error crítico durante la inferencia multimodal:", e)