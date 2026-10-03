# Resultados finales del TFG

Este directorio reúne las figuras de resultados finales incluidas en el capítulo 7 de `TFG_Jacobo.pdf`. Las imágenes se extrajeron del PDF sin modificar.

## YOLO - primera metodología

Carpeta: `YOLO_Metodologia_1/`

- Figura 7.1: matriz de confusión normalizada del detector binario.
- Recall indicado en la memoria: buen estado **0,80** y mal estado **0,85**.
- Parámetros de la última prueba (cap. 6): 200 épocas, `imgsz=600`, optimizador Adam, `batch=-1`, `lr0=0.0005`, `lrf=0.01`, `weight_decay=0.0001` y `patience=70`.

## Keras - primera metodología

Carpeta: `Keras_Metodologia_1/`

- Figura 7.2: matriz de confusión.
- Figura 7.3: informe de clasificación.
- Recall por clase: canker **0,97**; greening **0,97**; manchas negras **0,96**; moho verde **0,99**; otros defectos **1,00**; scab **0,99**; trips **0,90**.
- Exactitud global del informe: **0,97** (1.617 muestras).
- Parámetros finales (cap. 6): `IMG_SIZE=512`, `BATCH_SIZE=16`, `EPOCHS=200`, `PATIENCE=70`.

## YOLO - segunda metodología

Carpeta: `YOLO_Metodologia_2/`

- Figura 7.4: matriz de confusión normalizada multiclase.
- Recall por clase reportado en las conclusiones: sanas **0,69**; canker **0,60**; greening **1,00**; manchas negras **0,33**; moho verde **0,83**; otros defectos **1,00**; scab **1,00**; trips **0,94**.
- Macro-recall calculado en la memoria: **0,7988**.
- Parámetros de la última prueba (cap. 6): 200 épocas, `imgsz=600`, optimizador Adam, `batch=-1`, `lr0=0.0005`, `lrf=0.01`, `weight_decay=0.0001` y `patience=70`.


