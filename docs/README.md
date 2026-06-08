# Documentación del proyecto

Índice de documentación técnica del repositorio **Child Mind Institute — Problematic Internet Use** (Proyecto I de innovación tecnológica en IA, MIAA, Universidad Icesi).

El README general del repositorio está en [`../README.md`](../README.md) (objetivo, equipo, estructura y enlaces a Kaggle).

---

## Guía rápida: ¿qué leer primero?

| Si necesitas… | Ve a… |
|---------------|--------|
| Instalar el entorno y colocar los CSV de Kaggle | [instalacion.md](instalacion.md) |
| Entender el pipeline, modelos y comandos CLI | [arquitectura.md](arquitectura.md) |
| Métricas y limitaciones | [limitaciones_modelo.md](limitaciones_modelo.md) |
| Notebook definitivo de entrega (submission) | [06_modelo_final_reduced.ipynb](../notebooks/06_modelo_final_reduced.ipynb) |
| Ejecutar todo el flujo de modelado | [Reproducibilidad](#reproducibilidad) (abajo) |

---

## Documentos

| Documento | Contenido |
|-----------|-----------|
| [instalacion.md](instalacion.md) | Requisitos de Python, `pip`/`conda`, ubicación de `train.csv`/`test.csv`, CLI, tests y orden de notebooks |
| [arquitectura.md](arquitectura.md) | Flujo de datos, catálogo de 14 modelos, regresión ordinal, métricas, comandos y artefactos |
| [limitaciones_modelo.md](limitaciones_modelo.md) | Desbalance de la clase severa (`sii = 3`), métricas OOF por clase, frase sugerida |
|

---

## Notebooks (orden sugerido)

Ejecutar desde la raíz del proyecto (`jupyter notebook notebooks/`). Cada notebook usa `%run init_notebook.py` en las primeras celdas.

| Notebook | Contenido |
|----------|-----------|
| `00_introduccion_y_carga` | Contexto del proyecto y carga inicial |
| `01_comprension_calidad_target` | Calidad de datos, target `sii`, desbalance, data leakage PCIAT |
| `02_preparacion_y_eda` | Limpieza y análisis exploratorio |
| `03_modelado_evaluacion` | Modelado baseline (CV exploratoria 5-fold) |
| `04_mejoras_qwk_ensemble` | Regresión ordinal, ensemble y mejoras orientadas a QWK |
| `05_modelos_finales_y_comparacion` | Síntesis: comparación, tuning, diseño 2+3, diagnóstico OOF |
| **`06_modelo_final_reduced`** | **Notebook reducido de entrega** (modelo ganador → `submission.csv`) |

Los notebooks `04` y `05` leen artefactos de `experiments/results/` generados por el CLI. El **`06`** puede ejecutarse solo con `train.csv`/`test.csv` (crea metadatos mínimos) o reutilizar artefactos del repo para las mismas cifras del informe.

---

## Artefactos principales (`experiments/`)

Generados por el CLI. Los CSV/JSON/figuras en `results/` y los checkpoints de producción en `checkpoints/` están versionados (ver [`checkpoints/README.md`](../experiments/checkpoints/README.md)).

| Ruta | Descripción |
|------|-------------|
| `results/all_models_cv.csv` | **Métrica principal para el informe:** QWK medio ± std (CV 10-fold) |
| `results/all_models_holdout.csv` | Métricas en holdout 80/20 (comparación rápida) |
| `results/tuning_best_params.json` | Hiperparámetros del ganador por CV |
| `results/design_merge_classes_23.csv` | Experimento 4 clases vs fusión 2+3 |
| `results/final_model_meta.json` | Modelo final, umbrales ordinales, flags de configuración |
| `results/oof_report_summary.json` | QWK OOF global + bloque `limitations` |
| `results/per_class_metrics_oof.csv` | Precision, recall, F1 por clase (OOF) |
| `results/submission.csv` | Predicciones para Kaggle (`predict`) |
| `results/figures/confusion_matrix_oof.png` | Matriz de confusión honesta (OOF) |
| `results/figures/model_comparison_qwk_all.png` | Comparación QWK de todos los modelos |
| `results/figures/permutation_importance_oof.png` | Variables más relevantes |
| `checkpoints/final_model.joblib` | Modelo de producción (submission / `predict`) |
| `checkpoints/tuned_model.joblib` | Mejor pipeline tras `tune` (80/20; referencia, no submission) |

---

## Reproducibilidad

Flujo completo (desde la raíz del proyecto, con `data/raw/train.csv` y `test.csv` en local):

```bash
pip install -r requirements.txt   # Instala dependencias (pandas, sklearn, lightgbm, etc.).
python -m src.main check   # Carga train.csv, limpia datos y verifica shape 2736×46.
python -m src.main train   # Entrena 10 clasificadores (dummy, RF, SVM, LightGBM, …). Split 80/20 + CV 10-fold. => all_models_holdout.csv, all_models_cv.csv, gráfico comparativo
python -m src.main train-ensemble   # Entrena 4 modelos ordinales (3 regresores + vote). Añade filas a los mismos CSV. => Filas lgbm_regressor, xgb_regressor, etc.
python -m src.main tune   # Toma el mejor por CV y busca hiperparámetros con RandomizedSearchCV (métrica QWK). => tuning_best_params.json, tuned_model.joblib
python -m src.main design   # Prueba si fusionar clases 2+3 mejora QWK vs 4 clases. => design_merge_classes_23.csv
python -m src.main fit-final   # Entrena el modelo final (mejor modelo + params del tune) y lo guarda para entrega. => final_model.joblib, final_model_meta.json
python -m src.main report-oof  # figuras honestas OOF para el informe
python -m src.main predict     # submission.csv para Kaggle (requiere fit-final)
python -m unittest discover -s tests -v
```

Con conda: `conda env create -f environment.yml` y `conda activate proyecto-innovacion-tecnologica`.

Detalle de cada comando: [arquitectura.md](arquitectura.md) y [instalacion.md](instalacion.md).

---

## Datos de Kaggle

Los CSV de la competición **no deben subirse a Git** (reglas Kaggle y tamaño). Colocarlos en `data/raw/` según [instalacion.md](instalacion.md). El repositorio solo incluye `data/raw/.gitkeep` como marcador de ruta.

---

## Métricas para el informe

| Uso | Fuente |
|-----|--------|
| Comparación de modelos | `all_models_cv.csv` (QWK CV 10-fold) |
| Modelo elegido y QWK OOF | `oof_report_summary.json` |
| Limitaciones (clase 3) | `limitaciones_modelo.md` + `per_class_metrics_oof.csv` |
| Experimento diseño 2+3 | `design_merge_classes_23.csv` |
| Figuras | `experiments/results/figures/*_oof.png`, `model_comparison_qwk_all.png` |

Modelo final reportado: **`lgbm_regressor`** (QWK CV ≈ 0,413; QWK OOF ≈ 0,433 en la corrida documentada).
