"""Despliega en Render una imagen ya publicada en GitHub Container Registry y la verifica.

Render ejecuta el servicio a partir de la imagen que construyó y probó el CD. El script llama
al deploy hook del servicio con el parámetro imgURL, que fija la etiqueta exacta de la imagen
(la del modelo que se despliega), y luego consulta /salud hasta que el servicio responde con
esa versión. El plan gratuito se duerme tras 15 minutos sin tráfico y tarda cerca de un minuto
en despertar, así que la espera tolera esos arranques.

Uso (con RENDER_HOOK y RENDER_URL definidos):
  python src/desplegar_render.py --imagen ghcr.io/usuario/mlops-phishing-api:v8-abc1234 --version 8 \
      --exportado 2026-11-21T23:34:30+00:00
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def url_del_hook(hook: str, imagen: str) -> str:
    """Agrega imgURL, codificado, a la URL del deploy hook (que ya trae su parámetro key)."""
    separador = "&" if "?" in hook else "?"
    return f"{hook}{separador}imgURL={urllib.parse.quote(imagen, safe='')}"


def disparar(hook: str, imagen: str) -> int:
    """Inicia el despliegue. Render responde 200 (iniciado) o 202 (otro despliegue en curso)."""
    req = urllib.request.Request(url_del_hook(hook, imagen), method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status


def sirve_la_version(salud: dict, version: str, exportado: str | None = None) -> bool:
    """True si /salud muestra la versión pedida y, si se indica, la misma exportación del modelo.

    La versión sola no basta, porque mientras arranca la imagen nueva (o si su despliegue
    falla) Render sigue sirviendo la anterior, que puede llevar la misma versión. La marca
    exportado_utc cambia en cada construcción del CD e identifica la imagen recién publicada.
    """
    if salud.get("estado") != "ok" or str(salud.get("version_modelo")) != str(version):
        return False
    return exportado is None or salud.get("exportado_utc") == exportado


def esperar_version(url_servicio: str, version: str, minutos: int, exportado: str | None = None) -> dict:
    """Consulta /salud hasta que el servicio sirve la imagen desplegada o se agota el tiempo."""
    limite = time.time() + 60 * minutos
    ultimo: dict | str = "sin respuesta"
    while time.time() < limite:
        try:
            with urllib.request.urlopen(f"{url_servicio.rstrip('/')}/salud", timeout=60) as r:
                ultimo = json.load(r)
            print(f"  /salud: versión {ultimo.get('version_modelo')}, estado {ultimo.get('estado')}, "
                  f"exportado {ultimo.get('exportado_utc')}", flush=True)
            if sirve_la_version(ultimo, version, exportado):
                return ultimo
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            ultimo = str(e)
            print(f"  /salud todavía no responde ({e})", flush=True)
        time.sleep(20)
    raise RuntimeError(f"El servicio no sirvió la imagen nueva (versión {version}, exportada {exportado}) "
                       f"a tiempo; última respuesta: {ultimo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagen", required=True, help="referencia completa de la imagen con su etiqueta")
    ap.add_argument("--version", required=True, help="versión del modelo que debe servir el servicio")
    ap.add_argument("--exportado", required=True,
                    help="exportado_utc del version.json de la imagen, que la distingue de una anterior")
    ap.add_argument("--minutos", type=int, default=15, help="espera máxima del despliegue")
    args = ap.parse_args()

    hook, url_servicio = os.environ.get("RENDER_HOOK"), os.environ.get("RENDER_URL")
    if not hook or not url_servicio:
        sys.exit("Faltan RENDER_HOOK o RENDER_URL.")
    estado = disparar(hook, args.imagen)
    print(f"[desplegar] Render aceptó el despliegue de {args.imagen} (HTTP {estado}); esperando ...", flush=True)
    salud = esperar_version(url_servicio, args.version, args.minutos, args.exportado)
    print(f"[desplegar] En línea: {url_servicio} sirve la versión {salud['version_modelo']}")
    salida = os.environ.get("GITHUB_OUTPUT")
    if salida:
        with open(salida, "a", encoding="utf-8") as f:
            f.write(f"url={url_servicio}\nversion={args.version}\n")


if __name__ == "__main__":
    main()
