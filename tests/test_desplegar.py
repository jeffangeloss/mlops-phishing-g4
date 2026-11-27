"""Pruebas del despliegue en Render (sin red)."""
from urllib.parse import parse_qs, urlsplit

from desplegar_render import sirve_la_version, url_del_hook

HOOK = "https://api.render.com/deploy/srv-abc123?key=secreta"
IMAGEN = "ghcr.io/jeffangeloss/mlops-phishing-api:v8-41cafef"


def test_el_hook_lleva_la_imagen_codificada_y_conserva_la_clave():
    url = url_del_hook(HOOK, IMAGEN)
    q = parse_qs(urlsplit(url).query)
    assert q["key"] == ["secreta"]
    assert q["imgURL"] == [IMAGEN]
    assert "ghcr.io%2Fjeffangeloss%2Fmlops-phishing-api%3Av8-41cafef" in url


def test_el_hook_sin_parametros_usa_signo_de_pregunta():
    assert url_del_hook("https://api.render.com/deploy/srv-abc", IMAGEN).startswith(
        "https://api.render.com/deploy/srv-abc?imgURL=")


def test_solo_cuenta_como_desplegado_si_sirve_la_misma_version():
    assert sirve_la_version({"estado": "ok", "version_modelo": "8"}, "8")
    assert sirve_la_version({"estado": "ok", "version_modelo": 8}, "8")
    assert not sirve_la_version({"estado": "ok", "version_modelo": "7"}, "8")
    assert not sirve_la_version({"estado": "sin_modelo"}, "8")


def test_la_misma_version_de_una_imagen_anterior_no_cuenta_como_desplegada():
    # Render sigue sirviendo la imagen anterior mientras arranca la nueva (o si el despliegue
    # falla), y las dos pueden llevar la misma versión del modelo. Solo la marca de exportación
    # distingue la imagen que se acaba de construir.
    anterior = {"estado": "ok", "version_modelo": "8", "exportado_utc": "2026-11-21T17:33:17+00:00"}
    nueva = {"estado": "ok", "version_modelo": "8", "exportado_utc": "2026-11-21T23:34:30+00:00"}
    assert not sirve_la_version(anterior, "8", "2026-11-21T23:34:30+00:00")
    assert sirve_la_version(nueva, "8", "2026-11-21T23:34:30+00:00")
