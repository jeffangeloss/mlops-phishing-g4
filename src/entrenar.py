"""Etapa 2 del pipeline: entrenamiento, seguimiento en MLflow y registro como candidata.

Qué hace
  1. Comprueba que los datos sean los mismos que aprobó G1 (misma huella SHA-256).
  2. Parte PhiUSIIL 80/20 estratificado con semilla 42 y, en la versión 2 del dataset,
     agrega las URL legítimas con ruta solo a la partición de entrenamiento.
  3. Entrena un Pipeline de scikit-learn que recibe la URL: el extractor compartido
     (src/extractor.py) calcula las 20 características léxicas, la preparación se
     ajusta SOLO con la partición de entrenamiento y luego entrena
     CatBoostClassifier(verbose=0, random_state=42). El servicio usa el mismo Pipeline.
  4. Registra en MLflow parámetros, métricas con phishing como clase positiva,
     matriz de confusión, huella de datos, commit, entorno, ficha de modelo y modelo.
  5. Crea una nueva versión en el registro de modelos con el alias "candidata".

Uso:  python src/entrenar.py
"""
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

# comun va primero porque configura variables de entorno que MLflow lee al importarse.
from comun import (ALIAS_CANDIDATA, RAIZ, cargar_contrato, cargar_params, caracteristicas, commit_git,
                   configurar_mlflow, datos_entrenamiento, entorno, escribir_json, huella_datos,
                   metricas_phishing, origen_ejecucion, prob_phishing, responsable,
                   salir_con_error)

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from sklearn.model_selection import train_test_split  # noqa: E402

from extractor import ExtractorURL  # noqa: E402
from modelo import ModeloConUmbral, curva_de_umbrales, elegir_umbral_segun  # noqa: E402

SALIDA = RAIZ / "resultados" / "entrenamiento.json"
CODIGO_EXTRACTOR = RAIZ / "src" / "extractor.py"
CODIGO_MODELO = RAIZ / "src" / "modelo.py"


def construir_pipeline(params: dict) -> Pipeline:
    """Extractor + preparación + modelo en un solo objeto, que recibe la URL.

    El extractor es el primer paso, así que el modelo registrado calcula sus propias
    características y el servicio no puede calcularlas de otra forma. Sus tablas de
    referencia (TLD y frecuencia de caracteres) también se aprenden solo con entrenamiento.
    Al llamar a fit() con la partición de entrenamiento, el imputador aprende sus
    medianas solo de esas filas: la partición de prueba no influye en la
    preparación (en el cuaderno heredado el escalado y la selección se ajustaban
    con todo el conjunto, lo que filtraba información de la prueba).
    PhiUSIIL no trae nulos, así que el imputador no cambia los datos de
    entrenamiento; protege al servicio si algún día recibe un valor vacío.
    """
    cols = caracteristicas()
    preparacion = ColumnTransformer(
        [("imputar_mediana", SimpleImputer(strategy="median"), cols)],
        remainder="drop", verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    modelo = CatBoostClassifier(
        verbose=params["modelo"]["verbose"],
        random_state=params["modelo"]["semilla"],
        allow_writing_files=False,  # no crea la carpeta catboost_info; no cambia el modelo
    )
    extractor = ExtractorURL(prob_tld_desconocido=params["extractor"]["prob_tld_desconocido"])
    return Pipeline([("extractor", extractor), ("preparacion", preparacion), ("modelo", modelo)])


def ficha_de_modelo(nombre, metricas, huella, commit, ent, params, filas_extra=0, umbral=0.5) -> str:
    """Ficha de modelo breve (Mitchell et al., 2019), como pide la sección 4.4 del informe."""
    return f"""# Ficha de modelo: {nombre}

## Uso previsto
Clasificar una URL como phishing o legítima a partir de 20 características léxicas
de la cadena de la URL, antes de que el usuario abra el enlace. No descarga la página.
El modelo recibe la URL: el extractor compartido (src/extractor.py) calcula las
características dentro del propio Pipeline, igual en el entrenamiento y en el servicio.

## Datos
- Conjunto: versión {params['datos'].get('version', 1)} (huella SHA-256 {huella}). PhiUSIIL más
  {filas_extra} URL legítimas con ruta agregadas solo al entrenamiento (versión 2)
- Prueba: la partición original de PhiUSIIL, sin URL agregadas
- Partición: {int((1 - params['particion']['prueba']) * 100)}/{int(params['particion']['prueba'] * 100)} estratificada, semilla {params['particion']['semilla']}
- Clase positiva: phishing (label 0)

## Métricas en la prueba original (phishing como clase positiva)
- Umbral de decisión: {umbral} (elegido en una partición de validación del entrenamiento)
- Recall de phishing: {metricas['recall_phishing']:.4f}
- Precisión de phishing: {metricas['precision_phishing']:.4f}
- F1 de phishing: {metricas['f1_phishing']:.4f}
- Matriz de confusión: VP={metricas['vp_phishing_detectado']}, FN={metricas['fn_phishing_no_detectado']}, FP={metricas['fp_legitima_bloqueada']}, VN={metricas['vn_legitima_aceptada']}
- Tasa de URL legítimas bloqueadas fuera de distribución: ver la compuerta G4 en las etiquetas de la versión

## Linaje
- Commit: {commit}
- Entorno: Python {ent['python']}, catboost {ent['catboost']}, scikit-learn {ent['scikit-learn']}, pandas {ent['pandas']}

## Advertencias
- Sesgo de fuentes: las URL legítimas y las de phishing de PhiUSIIL provienen de fuentes
  distintas; el modelo puede aprender el origen de los datos (por ejemplo, la presencia
  de ruta) y fallar con URL legítimas con ruta o con TLD desconocidos.
- Métricas medidas solo en la prueba original, no en tráfico real.
- Las características salen del extractor y no del CSV, cuyos valores de longitud y de
  conteo dependen de la fuente de cada URL. URLCharProb usa una tabla de frecuencias
  aprendida con las URL de entrenamiento (ver docs/concordancia_extractor.md).
"""


def main():
    t_inicio = time.perf_counter()
    params = cargar_params()

    # 1. El CSV debe ser exactamente el que aprobó G1.
    huella_json = RAIZ / "data" / "huella.json"
    if not huella_json.exists():
        salir_con_error("No existe data/huella.json. Ejecuta antes: python src/datos.py")
    g1 = json.loads(huella_json.read_text(encoding="utf-8"))
    if g1.get("estado") != "aprobado":
        salir_con_error("La compuerta G1 rechazó los datos; revisa data/huella.json.")
    huella = huella_datos(params)
    if huella != g1["sha256"]:
        salir_con_error("Los datos cambiaron desde que G1 los aprobó. Vuelve a ejecutar src/datos.py.")

    # 2. Datos y partición. En la versión 2, las URL legítimas con ruta van solo al entrenamiento.
    X_ent, X_prueba, y_ent, y_prueba = datos_entrenamiento(params)
    filas_extra = int(g1.get("filas_legitimas_con_ruta", 0))

    commit, cambios = commit_git()
    ent = entorno()
    cliente = configurar_mlflow(params)
    nombre = params["modelo"]["nombre_registro"]

    # 3. Validación para el umbral: una parte del ENTRENAMIENTO (nunca de la prueba),
    #    estratificada por etiqueta y por origen, para que incluya URL legítimas con ruta.
    cfg_u = params["umbral"]
    es_extra = pd.Series(np.arange(len(X_ent)) >= len(X_ent) - filas_extra)
    estratos = y_ent.astype(str).values + es_extra.astype(str).values
    X_fit, X_val, y_fit, y_val, _, extra_val = train_test_split(
        X_ent, y_ent, es_extra, test_size=cfg_u["validacion"], stratify=estratos,
        random_state=params["particion"]["semilla"])

    with mlflow.start_run(run_name=f"catboost-20lex-{datetime.now():%Y%m%d-%H%M%S}") as run:
        # 4. Entrenamiento: fit solo con la parte de entrenamiento que no es validación.
        pipe = construir_pipeline(params)
        t0 = time.perf_counter()
        pipe.fit(X_fit, y_fit)
        t_ajuste = time.perf_counter() - t0

        # 5. Umbral de decisión elegido en validación y guardado dentro del modelo.
        pos = cargar_contrato()["etiqueta"]["clase_positiva"]
        candidatos = np.round(np.arange(cfg_u["minimo"], cfg_u["maximo"] + 1e-9, cfg_u["paso"]), 4)
        p_val = ModeloConUmbral.desde_ajustado(pipe, 0.5, pos).prob_positiva(X_val)
        # Si no hay legítimas adicionales, usa las legítimas de PhiUSIIL en validación.
        legitimas_val = extra_val.to_numpy() if extra_val.any() else y_val.to_numpy() != pos
        curva = curva_de_umbrales(p_val, y_val.to_numpy() == pos, legitimas_val, candidatos)
        eleccion = elegir_umbral_segun(curva, cfg_u, params["compuertas"]["recall_phishing_min"])
        modelo = ModeloConUmbral.desde_ajustado(pipe, eleccion["umbral"], pos)

        # 6. Evaluación en la prueba original, phishing como positiva, con el umbral elegido.
        y_pred = modelo.predict(X_prueba)
        metricas = metricas_phishing(y_prueba, y_pred)
        # Huella de las probabilidades redondeadas a 12 decimales para comparar corridas.
        probs = np.round(prob_phishing(modelo, X_prueba), 12)
        huella_pred = hashlib.sha256(probs.tobytes()).hexdigest()

        # Parámetros: partición, modelo y entorno (el entorno es parte del linaje).
        mlflow.log_params({
            "caracteristicas": len(caracteristicas()),
            "proporcion_prueba": params["particion"]["prueba"],
            "semilla_particion": params["particion"]["semilla"],
            "estratificada": params["particion"]["estratificada"],
            "version_dataset": params["datos"].get("version", 1),
            "filas_legitimas_con_ruta": filas_extra,
            "filas_entrenamiento": len(X_fit),
            "filas_validacion": len(X_val),
            "umbral_decision": eleccion["umbral"],
            "umbral_criterio": eleccion["criterio"],
            "filas_prueba": len(X_prueba),
            "preparacion": "SimpleImputer(median) ajustado solo con entrenamiento",
            "entrada_modelo": "URL",
            "extractor": "src/extractor.py, tablas de TLD y de caracteres ajustadas con entrenamiento",
            "extractor_prob_tld_desconocido": params["extractor"]["prob_tld_desconocido"],
            "modelo_clase": "CatBoostClassifier",
            "modelo_random_state": params["modelo"]["semilla"],
            "modelo_verbose": params["modelo"]["verbose"],
            **{f"entorno_{k.replace('-', '_')}": v for k, v in ent.items()},
        })
        mlflow.log_metrics({**metricas, "segundos_ajuste": t_ajuste,
                            "recall_phishing_validacion": eleccion["recall_validacion"],
                            "legitimas_con_ruta_bloqueadas_validacion": eleccion["bloqueo_validacion"]})
        mlflow.set_tags({
            "huella_datos_sha256": huella,
            "csv": g1["csv"],
            "commit": commit or "sin_git",
            "codigo_con_cambios_sin_commit": str(cambios).lower(),
            "huella_predicciones": huella_pred,
            "entrenado_por": responsable(),
            "lote": os.environ.get("POC_LOTE", "manual"),
            "corrida": os.environ.get("POC_CORRIDA", "1"),
            **origen_ejecucion(),  # ci o local; en el CI, también el enlace a la ejecución
        })
        mlflow.log_dict({"filas_real_columnas_pred": ["phishing", "legitima"],
                         "matriz": [[metricas["vp_phishing_detectado"], metricas["fn_phishing_no_detectado"]],
                                    [metricas["fp_legitima_bloqueada"], metricas["vn_legitima_aceptada"]]]},
                        "matriz_confusion.json")
        mlflow.log_text(curva.to_csv(index=False), "curva_umbral.csv")
        mlflow.log_artifact(str(RAIZ / "requirements.txt"))
        mlflow.log_artifact(str(huella_json))
        with tempfile.TemporaryDirectory() as tmp:
            ficha = Path(tmp) / "ficha_modelo.md"
            ficha.write_text(ficha_de_modelo(nombre, metricas, huella, commit or "sin_git", ent, params, filas_extra,
                                             eleccion["umbral"]),
                             encoding="utf-8")
            mlflow.log_artifact(str(ficha))

        # 5. Modelo y versión en el registro de modelos.
        # MLflow 3 guarda los modelos de scikit-learn con skops, que solo acepta tipos
        # de una lista de confianza y no reconoce CatBoost. Se usa cloudpickle: es
        # seguro aquí porque el modelo solo se carga desde el registro propio del equipo.
        # El ejemplo de entrada es solo la columna URL, que es lo que envía el servicio, y
        # code_paths empaqueta el extractor junto al modelo para que pueda cargarse en otro lugar.
        info = mlflow.sklearn.log_model(
            sk_model=modelo, name="modelo", registered_model_name=nombre,
            input_example=pd.DataFrame({"URL": ["https://example.org/ejemplo"]}),
            serialization_format="cloudpickle",
            code_paths=[str(CODIGO_EXTRACTOR), str(CODIGO_MODELO)],
        )
        version = str(info.registered_model_version)
        cliente.set_model_version_tag(nombre, version, "huella_datos_sha256", huella)
        cliente.set_model_version_tag(nombre, version, "commit", commit or "sin_git")
        cliente.set_model_version_tag(nombre, version, "estado", "candidata")
        cliente.set_model_version_tag(nombre, version, "umbral_decision", str(eleccion["umbral"]))
        cliente.set_model_version_tag(nombre, version, "origen", origen_ejecucion()["origen"])
        cliente.set_registered_model_alias(nombre, ALIAS_CANDIDATA, version)

        duracion = time.perf_counter() - t_inicio
        mlflow.log_metric("duracion_corrida_s", duracion)

    resumen = {"run_id": run.info.run_id, "version": version, "duracion_corrida_s": round(duracion, 2),
               **{k: round(v, 6) if isinstance(v, float) else v for k, v in metricas.items()}}
    escribir_json(SALIDA, resumen)
    print(f"[entrenar] Versión {version} registrada como '{ALIAS_CANDIDATA}'. "
          f"Recall de phishing {metricas['recall_phishing']:.4f}. Duración {duracion:.1f} s.")
    if commit is None:
        print("[entrenar] Aviso: la carpeta no es un repositorio Git; el linaje queda sin commit.")
    elif cambios:
        print("[entrenar] Aviso: hay cambios sin commit en el código; el linaje queda incompleto.")


if __name__ == "__main__":
    main()
