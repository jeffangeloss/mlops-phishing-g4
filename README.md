# mlops-phishing

Implementación del proyecto final de MLOps del curso de DevOps (Universidad de Lima, ciclo 2026-2, Grupo 04). El repositorio lleva a la práctica la alternativa B que el grupo recomienda en la Exposición 01, con DVC, MLflow, GitHub Actions y FastAPI, sobre el clasificador CatBoost que detecta URL de phishing a partir de las 20 características léxicas del conjunto PhiUSIIL.

> **Borrador.** Por ahora el repositorio contiene el entorno de Python, la configuración de las pruebas, la plantilla de variables y la bitácora. Cada integrante sube, revisa y prueba la parte que le corresponde, y este README se completa cuando el pipeline queda integrado.

## Contenido

```text
mlops-phishing/
├── docs/bitacora.md      dificultades y decisiones técnicas, en la fecha en que ocurren
├── requirements*.txt     dependencias con versiones fijadas (Python 3.11)
├── pytest.ini            configuración de pytest (código en src/, pruebas en tests/)
└── .env.example          variables del servidor compartido de MLflow en DagsHub
```

## Puesta en marcha

El proyecto fija Python 3.11 porque catboost 1.2.8 y scikit-learn 1.5.2 son la combinación probada. Con [uv](https://docs.astral.sh/uv/) instalado, la preparación del entorno toma unos minutos.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate              # Windows: .venv\Scripts\Activate.ps1
uv pip install -r requirements-dev.txt
```

Las credenciales del servidor compartido de MLflow van en un archivo `.env`, que git ignora, con las variables de `.env.example`.

## Flujo de ramas (GitFlow)

El repositorio sigue GitFlow. Cada integrante trabaja en su propia rama y la integra a `develop` después de traer lo último del equipo, así que su rama siempre contiene el trabajo de los demás.

| Rama | Uso |
|---|---|
| `feature/<tarea>` | Una por integrante o por tarea, creada desde `develop` |
| `develop` | Integración y rama por defecto. Recibe las ramas `feature/*` |
| `main` | Producción. Recibe de `develop` la versión que se entrega |

```bash
git switch -c feature/mi-tarea origin/develop
# después de los commits
git fetch origin
git merge --no-edit origin/develop
git push -u origin feature/mi-tarea
git push origin HEAD:develop
```

Si GitHub rechaza el último push, `develop` ya tiene un commit que la rama no trae, y basta repetir el `git fetch` y el `git merge` antes de volver a subir.

## Convenciones del equipo

- Cada cambio llega a `develop` desde una rama `feature/*`, y cada integrante hace commits con su propia cuenta, porque la coevaluación se contrasta con el historial del repositorio.
- Toda dificultad y toda decisión técnica se anota el mismo día en `docs/bitacora.md`.
- Cada captura que va al informe se guarda en `evidencias/` con su código y su fecha.
- Las cifras del informe salen de los registros de MLflow, de GitHub Actions o de `resultados/`, nunca de una transcripción a mano.
