import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras.utils import image_dataset_from_directory

def evaluar_modelo_detallado():
    # --- Configuración ---
    IMG_SIZE = 128
    BATCH_SIZE = 16
    MODEL_PATH = 'C:/Users/jacob/Desktop/TFG Jacobo/Pruebas/versiones_Keras/modelo_cnn_enfermedades_128.h5'
    DATASET_DIR = 'C:/Users/jacob/Desktop/TFG Jacobo/Pruebas/predicciones/predict_malas_256_enfermedades'

    model = load_model(MODEL_PATH)
    
    # 1. Cargamos TODO el dataset de la carpeta sin split para evaluar todo
    # Importante: shuffle=False para que no se desordenen las etiquetas
    test_ds = image_dataset_from_directory(
        DATASET_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False  
    )
    
    class_names = test_ds.class_names
    
    # 2. Normalizar y preparar datos
    # Extraemos etiquetas reales (y_true) antes de normalizar
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    test_ds_norm = test_ds.map(lambda x, y: (x / 255.0, y))

    # 3. Obtener predicciones (probabilidades)
    print("Iniciando predicciones...")
    predictions = model.predict(test_ds_norm)
    y_pred = np.argmax(predictions, axis=1) # Convertir probabilidad a clase (0, 1, 2...)

    # --- REPORTE DE MÉTRICAS ---
    print("\n📊 REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- MATRIZ DE CONFUSIÓN ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicción del Modelo')
    plt.ylabel('Clase Real (Carpeta)')
    plt.title('Matriz de Confusión')
    plt.show()

    # Ejemplo de ver las probabilidades del primer elemento
    print(f"\nProbabilidades de la primera imagen: {predictions[1]}")
    print(f"Clase predicha: {class_names[y_pred[1]]}")

evaluar_modelo_detallado()