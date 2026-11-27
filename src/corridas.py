"""Ejecuta el pipeline varias veces y resume lo que pide la Tabla 9 del informe.

Cada corrida ejecuta las tres etapas de dvc.yaml (datos -> entrenar -> compuertas)
con la misma huella, el mismo commit, el mismo entorno y la misma semilla. Si la
carpeta tiene Git y DVC inicializados, usa `dvc repro --force`; si no, llama a los
mismos tres comandos directamente.

Al final escribe resultados/resumen_corridas.csv con una fila por corrida:
  - si repite las métricas de la corrida 1 (y sus predicciones, bit a bit),
  - si su versión tiene linaje completo y qué le falta,
  - el recall de phishing y la decisión de la compuerta G3,
  - la duración desde la lectura del CSV hasta el registro (y la del ajuste y la del pipeline entero).

Uso:  python src/corridas.py [--n 3]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from comun import RAIZ, cargar_params, configurar_mlflow, linaje_completo

METRICAS = ["recall_phishing", "precision_phishing", "f1_phishing", "vp_phishing_detectado",
            "fn_phishing_no_detectado", "fp_legitima_bloqueada", "vn_legitima_aceptada"]


def ejecutable_dvc(env: dict) -> str | None:
    if (RAIZ / ".dvc").is_dir() and (RAIZ / ".git").exists():
        return shutil.which("dvc", path=env["PATH"])
    return None


def ejecutar_corrida(env: dict, dvc: str | None) -> None:
    if dvc:
        comandos = [[dvc, "repro", "--force"]]
    else:
        comandos = [[sys.executable, "src/datos.py"], [sys.executable, "src/entrenar.py"],
                    [sys.executable, "src/compuertas.py", "evaluar"]]
    for c in comandos:
        print("$", " ".join(Path(c[0]).name if i == 0 else x for i, x in enumerate(c)), flush=True)
        subprocess.run(c, cwd=RAIZ, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description="Ejecuta el pipeline y resume las corridas registradas en MLflow.")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    if args.n < 1:
        ap.error("--n debe ser mayor o igual a 1")

    lote = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex}"
    env = dict(os.environ, POC_LOTE=lote)
    # El "python" que DVC invoca debe ser el del entorno virtual activo.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    dvc = ejecutable_dvc(env)
    print(f"Lote {lote}: {args.n} corridas con {'DVC' if dvc else 'los comandos directos (sin Git o sin DVC)'}")

    duraciones = {}
    for i in range(1, args.n + 1):
        print(f"\n===== Corrida {i} de {args.n} =====", flush=True)
        env["POC_CORRIDA"] = str(i)
        t0 = time.perf_counter()
        ejecutar_corrida(env, dvc)
        duraciones[str(i)] = time.perf_counter() - t0

    # Lectura de las corridas del lote desde MLflow (la fuente de verdad, no la consola).
    params = cargar_params()
    cliente = configurar_mlflow(params)
    nombre = params["modelo"]["nombre_registro"]
    exp = cliente.get_experiment_by_name(params["mlflow"]["experimento"])
    runs = cliente.search_runs([exp.experiment_id], filter_string=f"tags.lote = '{lote}'")
    runs = sorted(runs, key=lambda r: int(r.data.tags["corrida"]))
    if [int(r.data.tags["corrida"]) for r in runs] != list(range(1, args.n + 1)):
        ap.exit(1, f"El lote {lote} no contiene exactamente las {args.n} corridas esperadas en MLflow.\n")
    versiones = {mv.run_id: mv for mv in cliente.search_model_versions(f"name='{nombre}'")}

    filas, base = [], None
    for r in runs:
        m, t = r.data.metrics, r.data.tags
        mv = versiones.get(r.info.run_id)
        if mv is None:
            ap.exit(1, f"La corrida {r.info.run_id} no tiene una versión del modelo registrada.\n")
        completo, faltan = linaje_completo(cliente, nombre, mv.version)
        firma = tuple(m[k] for k in METRICAS) + (t["huella_predicciones"],)
        base = base or firma
        filas.append({
            "corrida": int(t["corrida"]),
            "version_modelo": mv.version,
            "run_id": r.info.run_id,
            "huella_datos_sha256": t["huella_datos_sha256"],
            "commit": t["commit"],
            "recall_phishing": m["recall_phishing"],
            "precision_phishing": m["precision_phishing"],
            "f1_phishing": m["f1_phishing"],
            "vp": int(m["vp_phishing_detectado"]), "fn": int(m["fn_phishing_no_detectado"]),
            "fp": int(m["fp_legitima_bloqueada"]), "vn": int(m["vn_legitima_aceptada"]),
            "huella_predicciones": t["huella_predicciones"],
            "repite_metricas_de_corrida_1": firma == base,
            "linaje_completo": completo,
            "linaje_faltante": "; ".join(faltan),
            "g3_decision": mv.tags.get("g3_decision", "sin_evaluar"),
            "decision_compuertas": mv.tags.get("compuertas_automaticas", "sin_evaluar"),
            "duracion_corrida_s": round(m["duracion_corrida_s"], 2),
            "duracion_ajuste_s": round(m["segundos_ajuste"], 2),
            "duracion_pipeline_s": round(duraciones[t["corrida"]], 2),
        })

    tabla = pd.DataFrame(filas)
    salida = RAIZ / "resultados" / "resumen_corridas.csv"
    salida.parent.mkdir(exist_ok=True)
    tabla.to_csv(salida, index=False)

    n = len(tabla)
    print(f"\nResumen guardado en {salida.relative_to(RAIZ)}")
    print(f"  Corridas que repiten sus métricas: {int(tabla.repite_metricas_de_corrida_1.sum())} de {n}")
    print(f"  Versiones con linaje completo:     {int(tabla.linaje_completo.sum())} de {n}")
    print(f"  Recall de phishing por corrida:    {', '.join(f'{v:.4f}' for v in tabla.recall_phishing)}")
    print(f"  Decisión de G3 por corrida:        {', '.join(tabla.g3_decision)}")
    print(f"  Duración media (CSV -> registro):  {tabla.duracion_corrida_s.mean():.1f} s")
    print(f"  Duración media del ajuste (fit):   {tabla.duracion_ajuste_s.mean():.1f} s")
    print(f"  Duración media del pipeline:       {tabla.duracion_pipeline_s.mean():.1f} s")


if __name__ == "__main__":
    main()
