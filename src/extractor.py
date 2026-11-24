"""Extractor compartido de las 20 características léxicas desde la cadena de la URL.

El mismo código calcula las características en el entrenamiento y en el servicio, porque
forma el primer paso del Pipeline que se registra en MLflow. Así el modelo recibe la URL
y no un vector calculado por otro programa, y entre entrenamiento y servicio no puede
aparecer un desajuste de cálculo.

Las definiciones siguen las del conjunto PhiUSIIL, comprobadas contra su CSV con
src/concordancia.py, con tres diferencias que se documentan en docs/concordancia_extractor.md:

- URLLength y los conteos de letras, dígitos y caracteres especiales usan la URL completa
  tal como llega. En el CSV, esos valores dependen de la fuente de cada URL (las legítimas
  aparecen con un carácter menos), un artefacto que el servicio no puede reproducir.
- URLCharProb usa una tabla de frecuencias de caracteres aprendida de las URL de
  entrenamiento, porque la tabla original de los autores no está publicada.
- La ofuscación cuenta cualquier secuencia %XX en hexadecimal, sin distinguir mayúsculas.

La extracción no descarga la página ni visita la URL: solo analiza el texto.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Orden de las 20 características, igual al de contrato_datos.yaml.
CARACTERISTICAS = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate", "TLDLegitimateProb",
    "URLCharProb", "TLDLength", "NoOfSubDomain", "HasObfuscation", "NoOfObfuscatedChar",
    "ObfuscationRatio", "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL",
    "DegitRatioInURL", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
]

_ESQUEMA = re.compile(r"^[a-z][a-z0-9+.\-]*://")
_WWW = re.compile(r"^www\.")
_IPV4 = re.compile(r"\d{1,3}(\.\d{1,3}){3}")
_OFUSCACION = re.compile(r"%[0-9a-f]{2}")
_LETRAS = re.compile(r"[a-z]")
_DIGITOS = re.compile(r"[0-9]")
_OTROS_ESPECIALES = re.compile(r"[^a-z0-9=?&]")
_RACHAS = (re.compile(r"[a-z]+"), re.compile(r"[0-9]+"), re.compile(r"[^a-z0-9]+"))


def partes_url(url: str) -> dict:
    """Normaliza la URL (minúsculas, sin espacios en los extremos) y separa sus partes."""
    u = str(url).strip().lower()
    resto = _ESQUEMA.sub("", u)                  # sin http:// ni https://
    host = resto.split("/")[0].split("?")[0].split("#")[0]
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    sin_www = _WWW.sub("", host)
    nucleo = sin_www[: -(len(tld) + 1)] if tld and sin_www.endswith("." + tld) else sin_www
    return {"url": u, "resto": resto, "cuerpo": _WWW.sub("", resto), "host": host,
            "tld": tld, "nucleo": nucleo}


def tasa_de_continuidad(texto: str) -> float:
    """Suma de la racha más larga de letras, de dígitos y de especiales, sobre la longitud."""
    if not texto:
        return 0.0
    rachas = sum(max((len(m) for m in patron.findall(texto)), default=0) for patron in _RACHAS)
    return rachas / len(texto)


def caracteristicas_de_url(url: str, prob_tld: dict, prob_caracter: dict,
                           prob_tld_desconocido: float = 0.0) -> dict:
    """Las 20 características de una URL, con las tablas de referencia que recibe."""
    p = partes_url(url)
    u, cuerpo, host, tld = p["url"], p["cuerpo"], p["host"], p["tld"]
    largo = max(len(u), 1)
    n_ofuscados = 3 * len(_OFUSCACION.findall(u))
    letras = len(_LETRAS.findall(cuerpo))
    digitos = len(_DIGITOS.findall(cuerpo))
    especiales = len(_OTROS_ESPECIALES.findall(cuerpo))
    prob_car = float(np.mean([prob_caracter.get(c, 0.0) for c in cuerpo])) if cuerpo else 0.0
    return {
        "URLLength": len(u),
        "DomainLength": len(host),
        "IsDomainIP": int(bool(_IPV4.fullmatch(host))),
        "CharContinuationRate": tasa_de_continuidad(p["nucleo"]),
        "TLDLegitimateProb": float(prob_tld.get(tld, prob_tld_desconocido)),
        "URLCharProb": prob_car,
        "TLDLength": len(tld),
        "NoOfSubDomain": max(host.count(".") - 1, 0),
        "HasObfuscation": int(n_ofuscados > 0),
        "NoOfObfuscatedChar": n_ofuscados,
        "ObfuscationRatio": n_ofuscados / largo,
        "NoOfLettersInURL": letras,
        "LetterRatioInURL": letras / largo,
        "NoOfDegitsInURL": digitos,
        "DegitRatioInURL": digitos / largo,
        "NoOfEqualsInURL": u.count("="),
        "NoOfQMarkInURL": u.count("?"),
        "NoOfAmpersandInURL": u.count("&"),
        "NoOfOtherSpecialCharsInURL": especiales,
        "SpacialCharRatioInURL": especiales / largo,
    }


class ExtractorURL(BaseEstimator, TransformerMixin):
    """Primer paso del Pipeline: de una columna URL a las 20 características.

    fit() aprende dos tablas de referencia solo con la partición de entrenamiento.
    La de TLD toma el valor TLDLegitimateProb que el conjunto asigna a cada TLD (un valor
    fijo por TLD, calculado por los autores fuera de las etiquetas). La de caracteres es la
    frecuencia de cada carácter en las URL de entrenamiento. transform() solo necesita la
    columna URL, que es lo único que recibe el servicio.
    """

    def __init__(self, columna_url: str = "URL", columna_tld: str = "TLD",
                 columna_prob_tld: str = "TLDLegitimateProb", prob_tld_desconocido: float = 0.0):
        self.columna_url = columna_url
        self.columna_tld = columna_tld
        self.columna_prob_tld = columna_prob_tld
        self.prob_tld_desconocido = prob_tld_desconocido

    def fit(self, X: pd.DataFrame, y=None):
        urls = X[self.columna_url].astype(str)
        conteo = Counter()
        for url in urls:
            conteo.update(partes_url(url)["cuerpo"])
        total = sum(conteo.values()) or 1
        self.prob_caracter_ = {c: n / total for c, n in conteo.items()}
        if self.columna_tld in X.columns and self.columna_prob_tld in X.columns:
            pares = X[[self.columna_tld, self.columna_prob_tld]].dropna()
            self.prob_tld_ = (pares.astype({self.columna_tld: str})
                              .groupby(self.columna_tld)[self.columna_prob_tld].first().to_dict())
        else:
            self.prob_tld_ = {}
        return self

    def transform(self, X) -> pd.DataFrame:
        urls = X[self.columna_url] if isinstance(X, pd.DataFrame) else pd.Series(X)
        filas = [caracteristicas_de_url(u, self.prob_tld_, self.prob_caracter_, self.prob_tld_desconocido)
                 for u in urls.astype(str)]
        indice = X.index if isinstance(X, pd.DataFrame) else None
        return pd.DataFrame(filas, columns=CARACTERISTICAS, index=indice).astype("float64")

    def get_feature_names_out(self, input_features=None):
        return np.array(CARACTERISTICAS, dtype=object)
