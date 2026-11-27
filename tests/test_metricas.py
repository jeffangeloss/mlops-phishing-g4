"""Criterios de aceptación de la HU-03 para métricas, partición y compuertas."""

import pandas as pd
import pytest

import comun


def test_metricas_usan_phishing_como_clase_positiva(monkeypatch):
    monkeypatch.setattr(comun, "cargar_contrato", lambda: {"etiqueta": {"clase_positiva": 0}})

    metricas = comun.metricas_phishing([0, 0, 1, 1, 1], [0, 1, 1, 1, 1])

    assert metricas == {
        "recall_phishing": 0.5,
        "precision_phishing": 1.0,
        "f1_phishing": pytest.approx(2 / 3),
        "vp_phishing_detectado": 1,
        "fn_phishing_no_detectado": 1,
        "fp_legitima_bloqueada": 0,
        "vn_legitima_aceptada": 3,
    }


def test_particion_es_reproducible_y_estratificada():
    params = comun.cargar_params()
    X = pd.DataFrame({"x": range(100)})
    y = pd.Series([0] * 40 + [1] * 60)

    X_ent, X_prueba, y_ent, y_prueba = comun.particionar(X, y, params)
    X_ent_2, X_prueba_2, y_ent_2, y_prueba_2 = comun.particionar(X, y, params)

    pd.testing.assert_frame_equal(X_ent, X_ent_2)
    pd.testing.assert_frame_equal(X_prueba, X_prueba_2)
    pd.testing.assert_series_equal(y_ent, y_ent_2)
    pd.testing.assert_series_equal(y_prueba, y_prueba_2)
    assert X_ent.index.equals(y_ent.index)
    assert X_prueba.index.equals(y_prueba.index)
    assert X_ent.index.intersection(X_prueba.index).empty
    assert len(X_prueba) == 20
    assert (y_ent == 0).mean() == (y_prueba == 0).mean() == 0.4


def test_metas_de_g3_g4_y_g5_definidas_antes_de_entrenar():
    compuertas = comun.cargar_params()["compuertas"]

    assert 0 < compuertas["recall_phishing_min"] <= 1
    assert 0 <= compuertas["legitimas_bloqueadas_max"] <= 1
    assert 0 <= compuertas["tolerancia_vs_vigente"] <= 1
