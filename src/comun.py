"""Funciones compartidas por todas las etapas del kit.

Reunir aquí la lectura de parámetros, la partición y las métricas garantiza que
entrenar.py, compuertas.py, servicio.py y deriva.py usen exactamente las mismas
reglas. Si cada script tuviera su propia copia, un cambio en uno solo rompería
la comparación entre etapas.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version as version_paquete
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Silencia un mensaje informativo de MLflow 3 que no aplica a este kit.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# Carpeta raíz del kit (la que contiene params.yaml), sin importar desde dónde se llame.
RAIZ = Path(__file__).resolve().parent.parent

# Alias del registro de modelos. MLflow 3 reemplaza los antiguos "stages" por alias.
ALIAS_CANDIDATA = "candidata"
ALIAS_PREPRODUCCION = "preproduccion"

# Archivos que definen el código de una corrida. Si alguno tiene cambios sin commit,
# el commit registrado no describe lo que realmente corrió y el linaje queda incompleto.
ARCHIVOS_DE_CODIGO = ["src", "params.yaml", "contrato_datos.yaml", "requirements.txt", "dvc.yaml"]

LIBRERIAS = ["pandas", "numpy", "scikit-learn", "catboost", "mlflow", "dvc", "fastapi"]


# ---------------------------------------------------------------- parámetros
def cargar_params() -> dict:
    with open(RAIZ / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_contrato() -> dict:
    with open(RAIZ / "contrato_datos.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def caracteristicas() -> list[str]:
    """Las 20 características léxicas, en el orden del contrato."""
    return list(cargar_contrato()["caracteristicas"].keys())


def columnas_entrada() -> list[str]:
    """Columnas que recibe el Pipeline: la URL, más el TLD y su probabilidad, con los que el
    extractor arma su tabla de referencia durante el entrenamiento. El servicio solo envía la URL."""
    entrada = cargar_contrato()["entrada"]
    return [entrada["url"], entrada["tld"], entrada["prob_tld"]]


def ruta_csv(params: dict) -> Path:
    """Ruta al CSV de PhiUSIIL; si es relativa, se toma desde la raíz del kit."""
    ruta = Path(params["datos"]["csv"]).expanduser()
    return ruta if ruta.is_absolute() else (RAIZ / ruta).resolve()


# ---------------------------------------------------------------- datos
def sha256_archivo(ruta: Path) -> str:
    """Huella de contenido: cambia si cambia un solo byte del archivo."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def leer_datos(params: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Lee la URL, su TLD y la probabilidad del TLD, más la etiqueta.

    Las 20 características no se toman del CSV: las calcula el extractor del Pipeline desde
    la URL, igual que en el servicio (ver src/extractor.py y docs/concordancia_extractor.md).
    """
    col_y = cargar_contrato()["etiqueta"]["columna"]
    cols = columnas_entrada()
    df = pd.read_csv(ruta_csv(params), usecols=cols + [col_y], encoding="utf-8-sig")
    return df[cols], df[col_y]


def ruta_legitimas(params: dict) -> Path | None:
    """Archivo de URL legítimas con ruta de la versión 2 del dataset, o None en la versión 1."""
    ruta = params["datos"].get("legitimas_con_ruta")
    if not ruta:
        return None
    ruta = Path(ruta).expanduser()
    return ruta if ruta.is_absolute() else (RAIZ / ruta).resolve()


def huella_datos(params: dict) -> str:
    """Huella del dataset que se entrena.

    En la versión 1 es la del CSV de PhiUSIIL. En la versión 2 combina esa huella con la del
    archivo de URL legítimas con ruta, así que cambia si cambia un byte de cualquiera de los dos.
    """
    principal = sha256_archivo(ruta_csv(params))
    extra = ruta_legitimas(params)
    if extra is None:
        return principal
    return hashlib.sha256(f"{principal}:{sha256_archivo(extra)}".encode()).hexdigest()


def leer_legitimas(params: dict) -> tuple[pd.DataFrame, pd.Series] | None:
    """URL legítimas con ruta de la versión 2, con las mismas columnas que leer_datos."""
    ruta = ruta_legitimas(params)
    if ruta is None:
        return None
    contrato = cargar_contrato()
    col_y, entrada = contrato["etiqueta"]["columna"], contrato["entrada"]
    df = pd.read_csv(ruta, usecols=[entrada["url"], entrada["tld"], col_y])
    # Sin probabilidad de TLD: el extractor arma su tabla de TLD solo con las filas de PhiUSIIL.
    df[entrada["prob_tld"]] = float("nan")
    return df[columnas_entrada()], df[col_y]


def datos_entrenamiento(params: dict):
    """Particiones de PhiUSIIL con las URL legítimas con ruta agregadas solo al entrenamiento.

    La partición de prueba es exactamente la de PhiUSIIL, así que las métricas de la versión 2
    se comparan con las de la versión 1 sobre las mismas filas.
    """
    X, y = leer_datos(params)
    X_ent, X_prueba, y_ent, y_prueba = particionar(X, y, params)
    extra = leer_legitimas(params)
    if extra is not None:
        X_ent = pd.concat([X_ent, extra[0]], ignore_index=True)
        y_ent = pd.concat([y_ent, extra[1]], ignore_index=True)
    return X_ent, X_prueba, y_ent, y_prueba


def particionar(X: pd.DataFrame, y: pd.Series, params: dict):
    """Partición 80/20 estratificada con semilla fija: siempre las mismas filas en prueba."""
    p = params["particion"]
    return train_test_split(
        X, y,
        test_size=p["prueba"],
        stratify=y if p["estratificada"] else None,
        random_state=p["semilla"],
    )


# ---------------------------------------------------------------- métricas
def metricas_phishing(y_real, y_pred) -> dict:
    """Métricas con phishing (label 0) como clase positiva.

    scikit-learn usa pos_label=1 por defecto, que aquí es la clase legítima. Por eso
    se fija pos_label=0: de lo contrario el recall reportado sería el de las URL
    legítimas (el error de orientación que señala el capítulo 3 del informe).
    """
    pos = cargar_contrato()["etiqueta"]["clase_positiva"]
    vp, fn, fp, vn = confusion_matrix(y_real, y_pred, labels=[pos, 1 - pos]).ravel()
    return {
        "recall_phishing": float(recall_score(y_real, y_pred, pos_label=pos)),
        "precision_phishing": float(precision_score(y_real, y_pred, pos_label=pos)),
        "f1_phishing": float(f1_score(y_real, y_pred, pos_label=pos)),
        # Matriz de confusión con phishing como positiva.
        "vp_phishing_detectado": int(vp),
        "fn_phishing_no_detectado": int(fn),
        "fp_legitima_bloqueada": int(fp),
        "vn_legitima_aceptada": int(vn),
    }


def prob_phishing(modelo, X: pd.DataFrame):
    """Probabilidad de la clase phishing (label 0), sin suponer el orden de las clases."""
    pos = cargar_contrato()["etiqueta"]["clase_positiva"]
    clases = list(modelo.classes_)
    return modelo.predict_proba(X)[:, clases.index(pos)]


# ---------------------------------------------------------------- entorno y código
def commit_git() -> tuple[str | None, bool]:
    """Devuelve (commit, hay_cambios_sin_commit) del código del kit, o (None, True) sin Git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
                             text=True, check=True).stdout.strip()
        cambios = subprocess.run(["git", "status", "--porcelain", "--", *ARCHIVOS_DE_CODIGO],
                                 cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()
        return sha, bool(cambios)
    except (OSError, subprocess.CalledProcessError):
        return None, True


def entorno() -> dict:
    """Versión de Python, sistema y librerías: parte del linaje de cada corrida."""
    datos = {"python": platform.python_version(), "sistema": platform.platform()}
    for lib in LIBRERIAS:
        try:
            datos[lib] = version_paquete(lib)
        except Exception:  # librería no instalada
            datos[lib] = "no_instalada"
    return datos


def responsable() -> str:
    """Quién entrena: la variable POC_USUARIO si existe, si no el usuario del sistema."""
    import getpass
    return os.environ.get("POC_USUARIO") or getpass.getuser()


def origen_ejecucion() -> dict:
    """Dónde corre el pipeline: en el CI (entorno de referencia) o en una máquina local.

    En GitHub Actions agrega la ejecución, su enlace, la rama y el usuario que la disparó,
    para que cada versión del modelo apunte a la página de la ejecución que la produjo.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return {"origen": "local"}
    servidor = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    ejecucion = os.environ.get("GITHUB_RUN_ID", "")
    return {
        "origen": "ci",
        "github_run_id": ejecucion,
        "github_run_url": f"{servidor}/{repo}/actions/runs/{ejecucion}",
        "github_ref": os.environ.get("GITHUB_REF_NAME", ""),
        "github_actor": os.environ.get("GITHUB_ACTOR", ""),
    }


# ---------------------------------------------------------------- MLflow
def mlflow_remoto() -> str | None:
    """URI del servidor MLflow compartido (DagsHub) si está configurado, o None."""
    return os.environ.get("MLFLOW_TRACKING_URI") or None


def configurar_mlflow(params: dict):
    """Apunta MLflow al servidor compartido o, si no está configurado, a la base SQLite local.

    El servidor compartido (DagsHub) es el que usan el CI y las corridas oficiales. Sin la
    variable MLFLOW_TRACKING_URI, cada integrante prueba con su propia base mlflow.db, que no
    consume las corridas del plan gratuito. Las credenciales se leen de las variables
    MLFLOW_TRACKING_USERNAME y MLFLOW_TRACKING_PASSWORD, nunca del código.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    cfg = params["mlflow"]
    remoto = mlflow_remoto()
    uri = remoto or f"sqlite:///{(RAIZ / cfg['base']).as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    cliente = MlflowClient()
    exp = cliente.get_experiment_by_name(cfg["experimento"])
    if exp is None:
        if remoto:
            # En el servidor compartido, el propio servidor decide dónde guarda los artefactos.
            cliente.create_experiment(cfg["experimento"])
        else:
            # En local, los artefactos (modelo, ficha, requisitos) quedan en ./mlruns.
            cliente.create_experiment(cfg["experimento"],
                                      artifact_location=(RAIZ / cfg["artefactos"]).as_uri())
    mlflow.set_experiment(cfg["experimento"])
    return cliente


def version_con_alias(cliente, nombre: str, alias: str):
    """Versión del registro que lleva el alias, o None si el alias no existe."""
    try:
        return cliente.get_model_version_by_alias(nombre, alias)
    except Exception:
        return None


def linaje_completo(cliente, nombre: str, version: str) -> tuple[bool, list[str]]:
    """Revisa que la versión tenga huella, commit, parámetros, entorno y ficha de modelo.

    Es la definición del indicador "versiones con linaje completo" de la Tabla 5.
    Devuelve (completo, lista_de_faltantes).
    """
    mv = cliente.get_model_version(nombre, version)
    run = cliente.get_run(mv.run_id)
    faltan = []
    if not mv.tags.get("huella_datos_sha256"):
        faltan.append("huella de datos")
    commit = run.data.tags.get("commit", "")
    if not commit or commit == "sin_git":
        faltan.append("commit")
    elif run.data.tags.get("codigo_con_cambios_sin_commit") == "true":
        faltan.append("commit limpio (había cambios sin commit)")
    if "semilla_particion" not in run.data.params or "modelo_random_state" not in run.data.params:
        faltan.append("parámetros")
    if "entorno_python" not in run.data.params:
        faltan.append("entorno")
    artefactos = [a.path for a in cliente.list_artifacts(mv.run_id)]
    if "ficha_modelo.md" not in artefactos:
        faltan.append("ficha de modelo")
    return (not faltan), faltan


def escribir_json(ruta: Path, datos: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


def salir_con_error(mensaje: str) -> None:
    print(f"\n[ERROR] {mensaje}\n", file=sys.stderr)
    sys.exit(1)
