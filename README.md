# 🍊 TFG_Naranjas

> **Estudio y aplicación de YOLO, redes convolucionales (CNN) y modelos de lenguaje (LLM) para la detección, clasificación y tratamiento de naranjas**

Este repositorio recoge el código, los conjuntos de datos, las pruebas y los resultados del Trabajo Fin de Grado de **Jacobo San Martín Sáez**, realizado en el Grado en Ingeniería Telemática. El proyecto explora cómo combinar modelos de visión por computador con un modelo de lenguaje para analizar imágenes de cítricos y ofrecer una recomendación asociada al diagnóstico.

## El proyecto en un vistazo

A partir de una imagen, el sistema busca localizar naranjas, identificar si presentan daños y clasificar la patología. La memoria compara dos formas de realizar ese análisis:

| Metodología | Flujo de análisis | Enfoque |
| --- | --- | --- |
| **1. PRIMERA METODOLOGÍA: YOLOv11 + Keras** | YOLOv11 detecta naranjas sanas o dañadas → se recortan las detecciones dañadas → Keras clasifica la enfermedad | Pipeline por etapas |
| **2. SEGUNDA METODOLOGÍA: YOLOv11** | Un modelo YOLOv11 detecta y clasifica directamente la naranja como sana o con una de siete categorías de defectos | Detección en una etapa |

En ambos flujos, la clase detectada puede enviarse a **Gemini** junto con la imagen y un prompt especializado para generar una recomendación de tratamiento. Esta integración se plantea como apoyo experimental; la memoria también señala que la llamada directa a un LLM tiene limitaciones para ofrecer asesoramiento agronómico especializado.

## Categorías analizadas

El clasificador multiclase contempla naranjas sanas y siete categorías de daños o enfermedades:

- Canker (cancrosis)
- Greening
- Manchas negras
- Moho verde
- Otros defectos
- Scab
- Trips

## Resultados principales

La memoria compara las metodologías mediante **macro-recall por clase**. Según los cálculos presentados, la primera metodología obtiene **0,8204** y la segunda **0,7988**. La arquitectura YOLO multiclase requiere menos etapas y menos experimentación, mientras que el pipeline YOLO + Keras logra el macro-recall superior en esta evaluación.

Los resultados dependen de los conjuntos de datos utilizados. En particular, la memoria atribuye parte de las dificultades del segundo YOLO al menor volumen y balance de imágenes; manchas negras obtiene un recall de **0,33**. Las matrices de confusión e informes finales, junto con el resumen de métricas, están en [`Proyecto/Resultados`](Proyecto/Resultados/README.md).

## Datos y evaluación

El trabajo emplea conjuntos etiquetados para los dos detectores YOLO y un conjunto organizado por carpetas para Keras. En la versión descrita en la memoria:

- El conjunto del primer YOLO contiene **1.861 imágenes** y **5.558 instancias anotadas**, equilibradas entre buen y mal estado.
- El conjunto del segundo YOLO contiene **906 imágenes** para la clasificación de naranjas sanas y las siete categorías de defectos.
- El conjunto de evaluación de Keras contiene **1.617 imágenes** organizadas por enfermedad.
- Los conjuntos de YOLO se distribuyeron en entrenamiento, validación y prueba; Keras se evaluó con un conjunto independiente descrito en la memoria.

Las particiones y cifras anteriores resumen el documento del TFG. Consulta los archivos disponibles en `Proyecto/Datasets` para ver el contenido presente en este repositorio.

## Tecnologías

- **Python** para el desarrollo de los scripts.
- **Ultralytics YOLOv11** para detección binaria y multiclase.
- **TensorFlow / Keras** para clasificar los recortes de naranjas dañadas.
- **OpenCV** para operaciones sobre imágenes y extracción de recortes.
- **Roboflow** para preparar y etiquetar conjuntos de datos.
- **Google Gemini API** para generar texto de apoyo al diagnóstico.

## Estructura del repositorio

```text
TFG_Naranjas/
├── README.md
├── TFG_Jacobo.pdf
└── Proyecto/
    ├── Codigo/       # Scripts de entrenamiento, predicción y utilidades
    ├── Datasets/     # Conjuntos de datos empleados
    ├── Pruebas/      # Predicciones y artefactos de evaluación
    └── Resultados/   # Figuras finales y resumen de métricas
```

Algunos scripts destacados de `Proyecto/Codigo`:

- `cnn_clasificador.py`: clasificador de enfermedades con Keras.
- `evaluar_modelo_cnn.py`: evaluación del clasificador.
- `extraer_malas_para_cnn.py`: preparación de recortes para el flujo YOLO + Keras.
- `data_augmentation.py`: aumento de datos.
- `predecir.py` y `detectar_clases.py`: utilidades de predicción y detección.
- `llm_gemini.py`: integración con Gemini.

## Ejecución y configuración

Los scripts están pensados para trabajar con rutas locales y los modelos/datasets preparados para el proyecto. Revisa las rutas y parámetros al inicio del script que quieras ejecutar y ajústalos a tu entorno.

La integración con Gemini requiere una clave de API. Configúrala localmente como `GOOGLE_API_KEY` (por ejemplo, mediante un archivo `.env` leído por `python-dotenv`) y no publiques credenciales ni las incluyas en commits.

## Memoria

La memoria completa, con el marco teórico, diseño, implementación, pruebas y conclusiones, está disponible en [`TFG_Jacobo.pdf`](TFG_Jacobo.pdf).

## IMPORTANTE

NO SE PRESENTA LA API KEY, NI EL ARCHIVO .env POR RAZONES DE PRIVACIDAD Y SEGURIDAD.
