# Línea base del flujo manual

La línea base documenta el flujo de la Exposición 01 para compararlo con el pipeline automatizado en el Capítulo 4. Usa el cuaderno `Modelo_Comparativo_PHILL.ipynb` del proyecto original y el conjunto PhiUSIIL. Registra quién hizo la medición, cuándo, el equipo y el sistema operativo, la versión del cuaderno y la huella del CSV.

## Procedimiento de medición

Inicia el cronómetro antes de cada paso y anota la duración, la cantidad de acciones manuales y la ruta de la evidencia. Registra por separado el tiempo de instalación y descarga. Conserva la salida del cuaderno y las capturas para respaldar las cifras.

| Paso | Procedimiento manual | Duración (min) | Acciones | Evidencia |
|---|---|---:|---:|---|
| Obtener datos | Descargar o copiar el CSV de PhiUSIIL y registrar su huella. | Pendiente | Pendiente | |
| Preparar entorno | Preparar el entorno del cuaderno y registrar las versiones instaladas. | Pendiente | Pendiente | |
| Entrenar | Ejecutar las celdas del cuaderno en orden y anotar el tiempo de ajuste. | Pendiente | Pendiente | |
| Evaluar | Guardar el resultado sobre la partición de prueba y la matriz de confusión. | Pendiente | Pendiente | |
| Guardar resultados | Exportar el modelo, conservar la salida y vincularla al cuaderno y al CSV. | Pendiente | Pendiente | |
| Publicar y revertir | Si se reconstruye la publicación manual, medir el tiempo hasta servir el modelo y el tiempo para recuperar la versión previa. Identificar ambas mediciones como reconstruidas. | Pendiente | Pendiente | |

## Comparación para el informe

- Entrenamiento y evaluación. Suma los tiempos de ambos pasos. Registra el tiempo de ajuste por separado para compararlo con `segundos_ajuste` de MLflow.
- Reproducibilidad y trazabilidad. Registra la huella de datos, la versión del cuaderno, el entorno y la semilla. Si se repite el entrenamiento, compara las métricas y la matriz de confusión sobre la misma partición.
- Métricas. Reporta recall, precisión y F1 con `label=0` (phishing) como clase positiva. Conserva verdaderos positivos, falsos negativos, falsos positivos y verdaderos negativos. Si el cuaderno reporta `label=1`, recalcula a partir de las predicciones o de la matriz original.
- Operación. Compara las acciones manuales, el versionado del modelo y los tiempos de publicación y reversión con los registros de los workflows cuando estén disponibles.

Las cifras manuales quedan pendientes de una medición del flujo original. Para el escenario automatizado de este repositorio, usa las corridas registradas en MLflow y el resumen que genera `src/corridas.py`.
