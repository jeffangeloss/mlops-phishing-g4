"""Modelo con umbral de decisión ajustado en una partición de validación.

CatBoost decide con un umbral fijo de 0,5 sobre la probabilidad de phishing. Ese valor no
tiene por qué equilibrar bien los dos errores que importan aquí: dejar pasar phishing
(recall, compuerta G3) y bloquear URL legítimas con ruta (compuerta G4). Este módulo elige
el umbral con datos de validación, separados del entrenamiento y nunca con la prueba, y lo
guarda dentro del propio modelo registrado, de modo que el servicio y las compuertas lo
aplican sin configurarlo aparte.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class ModeloConUmbral(BaseEstimator, ClassifierMixin):
    """Envuelve un Pipeline ya ajustado y clasifica como phishing cuando p >= umbral."""

    def __init__(self, pipeline=None, umbral: float = 0.5, clase_positiva: int = 0):
        self.pipeline = pipeline
        self.umbral = umbral
        self.clase_positiva = clase_positiva

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self

    @classmethod
    def desde_ajustado(cls, pipeline, umbral: float, clase_positiva: int = 0) -> "ModeloConUmbral":
        modelo = cls(pipeline, umbral, clase_positiva)
        modelo.classes_ = pipeline.classes_
        return modelo

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def prob_positiva(self, X) -> np.ndarray:
        return self.predict_proba(X)[:, list(self.classes_).index(self.clase_positiva)]

    def predict(self, X) -> np.ndarray:
        otra = next(c for c in self.classes_ if c != self.clase_positiva)
        return np.where(self.prob_positiva(X) >= self.umbral, self.clase_positiva, otra)

    @property
    def named_steps(self):
        """Pasos del Pipeline interno (el servicio usa el extractor para el monitoreo)."""
        return self.pipeline.named_steps


def curva_de_umbrales(prob_phishing, es_phishing, es_legitima_con_ruta, umbrales) -> pd.DataFrame:
    """Recall de phishing y tasa de legítimas con ruta bloqueadas para cada umbral candidato."""
    p = np.asarray(prob_phishing)
    phishing = np.asarray(es_phishing, dtype=bool)
    con_ruta = np.asarray(es_legitima_con_ruta, dtype=bool)
    filas = []
    for t in umbrales:
        bloquea = p >= t
        filas.append({
            "umbral": round(float(t), 4),
            "recall_phishing": float(bloquea[phishing].mean()) if phishing.any() else float("nan"),
            "legitimas_con_ruta_bloqueadas": float(bloquea[con_ruta].mean()) if con_ruta.any() else float("nan"),
        })
    return pd.DataFrame(filas)


def elegir_umbral_por_bloqueo(curva: pd.DataFrame, tope_bloqueo: float) -> dict:
    """Perfil equilibrado: el umbral más bajo (el de mayor recall) que, en validación, bloquea
    como máximo tope_bloqueo de las URL legítimas con ruta. Si ninguno lo logra, toma el que
    menos bloquea y lo informa, para que las compuertas decidan con su motivo."""
    cumplen = curva[curva["legitimas_con_ruta_bloqueadas"] <= tope_bloqueo]
    if not cumplen.empty:
        fila = cumplen.sort_values(["recall_phishing", "umbral"], ascending=[False, True]).iloc[0]
        criterio = (f"bloquea como máximo {tope_bloqueo:.0%} de las URL legítimas con ruta en validación, "
                    "con el mayor recall posible")
    else:
        fila = curva.sort_values(["legitimas_con_ruta_bloqueadas", "recall_phishing"],
                                 ascending=[True, False]).iloc[0]
        criterio = f"ningún umbral bloquea {tope_bloqueo:.0%} o menos; se toma el que menos bloquea"
    return {"umbral": float(fila["umbral"]), "recall_validacion": float(fila["recall_phishing"]),
            "bloqueo_validacion": float(fila["legitimas_con_ruta_bloqueadas"]), "criterio": criterio}


def elegir_umbral_segun(curva: pd.DataFrame, cfg_umbral: dict, piso_recall: float) -> dict:
    """Aplica el criterio que fija params.yaml (umbral.criterio)."""
    if cfg_umbral.get("criterio", "piso_recall") == "tope_bloqueo_validacion":
        return elegir_umbral_por_bloqueo(curva, cfg_umbral["tope_bloqueo"])
    return elegir_umbral(curva, piso_recall)


def elegir_umbral(curva: pd.DataFrame, piso_recall: float) -> dict:
    """Perfil de máxima detección: elige el umbral de la curva.

    Entre los umbrales que cumplen el piso de recall de G3, toma el que bloquea menos URL
    legítimas con ruta (a igualdad, el de mayor recall). Si ninguno cumple el piso, toma el de
    mayor recall y lo informa, para que las compuertas rechacen la versión con su motivo.
    """
    cumplen = curva[curva["recall_phishing"] >= piso_recall]
    if not cumplen.empty:
        fila = cumplen.sort_values(["legitimas_con_ruta_bloqueadas", "recall_phishing"],
                                   ascending=[True, False]).iloc[0]
        criterio = f"cumple el piso de recall {piso_recall} con el menor bloqueo de legítimas con ruta"
    else:
        fila = curva.sort_values(["recall_phishing", "legitimas_con_ruta_bloqueadas"],
                                 ascending=[False, True]).iloc[0]
        criterio = f"ningún umbral alcanza el piso de recall {piso_recall}; se toma el de mayor recall"
    return {"umbral": float(fila["umbral"]), "recall_validacion": float(fila["recall_phishing"]),
            "bloqueo_validacion": float(fila["legitimas_con_ruta_bloqueadas"]), "criterio": criterio}
