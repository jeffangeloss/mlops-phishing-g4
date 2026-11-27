#!/usr/bin/env bash
# Ejecuta tres corridas del pipeline y escribe resultados/resumen_corridas.csv
# (los datos de la Tabla 9 del informe).
# macOS o Linux:   ./run_pipeline.sh          (con el entorno virtual activado)
# Windows:         python src/corridas.py --n 3
set -euo pipefail
cd "$(dirname "$0")"
python src/corridas.py --n "${1:-3}"
