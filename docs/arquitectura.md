# Arquitectura del pipeline ML

Comparación de modelos para predecir uso problemático de internet (`sii`, ordinal 0–3) a partir de variables de actividad física y hábitos digitales. El flujo principal es reproducible desde `src/` y los notebooks.

## Estructura del código

| Módulo | Archivos | Rol |
|--------|----------|-----|
| `src/data/` | `load.py`, `preprocess.py`, `dataset.py`, `variable_families.py` | Carga, limpieza, split train/test |
| `src/models/` | `baseline.py`, `classical.py`, `boosting.py`, `ordinal.py`, `ensemble_regression.py`, `registry.py` | Definición y registro de pipelines |
| `src/training/` | `train.py`, `train_ensemble.py`, `tune.py`, `run_design_experiments.py`, `fit_final.py` | Entrenamiento, ensemble ordinal, tuning, diseño, modelo final |
| `src/evaluation/` | `metrics.py`, `thresholds.py`, `evaluate.py` | Métricas (QWK, F1, etc.), umbrales ordinales y análisis de errores |
| `src/main.py` | — | Punto de entrada CLI |

## Flujo de datos

```
data/raw/train.csv
    → src/data/load.py
    → src/data/preprocess.py (clean_for_modeling)
    → src/data/dataset.py (split 80/20 estratificado)
    → ColumnTransformer (build_preprocessor)
    → Pipeline por modelo
```

Shape esperado tras limpieza: **2736 × 46** (features derivadas, sin columnas `*-Season`; imputación numérica con `IterativeImputer` en el preprocessor).

Validación cruzada estratificada: **10 folds** (`CV_FOLDS` en `src/config.py`).

## Catálogo de modelos

| Clave | Nivel | Módulo |
|-------|-------|--------|
| `dummy` | Referencia | `baseline.py` |
| `logistic_regression` | Básico | `baseline.py` |
| `knn` | Básico | `classical.py` |
| `ridge_ordinal` | Básico | `classical.py` + `ordinal.py` |
| `decision_tree` | Básico | `baseline.py` |
| `naive_bayes` | Básico (opcional) | `classical.py` |
| `random_forest` | Intermedio | `baseline.py` |
| `svm` | Intermedio | `classical.py` |
| `lightgbm` | Avanzado | `boosting.py` |
| `xgboost` | Avanzado | `boosting.py` |
| `lgbm_regressor` | Avanzado | `ensemble_regression.py` |
| `xgb_regressor` | Avanzado | `ensemble_regression.py` |
| `extratrees_regressor` | Avanzado | `ensemble_regression.py` |
| `regression_vote` | Avanzado | `ensemble_regression.py` (mode vote) |

Registro central: `src/models/registry.py` → `get_all_model_pipelines()` (clasificadores) y `get_ensemble_regression_pipelines()` (regresión ordinal).

### Regresión ordinal y umbrales QWK

1. Cada regresor (`LGBMRegressor`, `XGBRegressor`, `ExtraTreesRegressor`) predice `sii` como valor continuo.
2. `ThresholdOrdinalClassifier` (`src/evaluation/thresholds.py`) optimiza tres umbrales en entrenamiento para maximizar QWK.
3. `regression_vote` entrena los tres regresores y aplica **mode vote** sobre las clases discretizadas.
4. CV del vote usa predicciones OOF por fold, umbrales optimizados en OOF y QWK agregado.

Umbrales del modelo final se guardan en `final_model_meta.json` cuando aplica.

## Métricas

- **Principal (tuning y comparación):** QWK (Cohen's kappa cuadrático)
- **Secundarias:** F1 macro, balanced accuracy, accuracy

Implementación: `src/evaluation/metrics.py`, `src/evaluation/evaluate.py`.

## Comandos y artefactos

| Comando | Script | Salidas |
|---------|--------|---------|
| `python -m src.main check` | — | Consola |
| `python -m src.main train` | `training/train.py` | `all_models_holdout.csv`, `all_models_cv.csv`, `figures/model_comparison_qwk.png` (baseline) |
| `python -m src.main train-ensemble` | `training/train_ensemble.py` | Añade filas ensemble a los CSV; `figures/model_comparison_qwk_all.png` |
| `python -m src.main tune` | `training/tune.py` | `tuning_best_params.json`, `tuned_model.joblib` (80/20; versioned, not used for predict) |
| `python -m src.main design` | `training/run_design_experiments.py` | `design_merge_classes_23.csv` |
| `python -m src.main fit-final` | `training/fit_final.py` | `final_model.joblib`, `final_model_meta.json` (full train; versioned) |
| `python -m src.main report-oof` | `evaluation/oof_report.py` | `confusion_matrix_oof.png`, `per_class_metrics_oof.csv`, `permutation_importance_oof.*`, `oof_report_summary.json` (incl. `limitations`) |
| `python -m src.main predict` | `evaluation/predict.py` | `submission.csv` (Kaggle) |
| `python -m src.main evaluate` | `evaluation/evaluate.py` | **Opcional.** Solo con `FINAL_FIT_ON_FULL_TRAIN=False` |

**Métrica para reportar:** `all_models_cv.csv` (CV 10-fold en `train` / `train-ensemble`). Tras `fit-final` no hay evaluación holdout honesta: el modelo ya vio todo el train etiquetado.

## Experimento de diseño

Fusión de clases 2 y 3 en una sola etiqueta (mapeo `3 → 2`) vs configuración original de 4 clases. El modelo final usa la variante con mejor QWK en holdout si el CSV de diseño existe.

## Métricas por fase

- `train` / `train-ensemble` / `tune` / `design`: comparación en CV o holdout 80/20 (sin fuga).
- `fit-final`: reentrena el mejor candidato con **todo** el train etiquetado.
- `report-oof`: predicciones out-of-fold (10 folds) del mismo pipeline/hiperparámetros; figuras honestas para el informe.
- `predict`: inferencia sobre `test.csv` con `final_model.joblib` → `submission.csv`.
- `evaluate`: omitir en el flujo normal; reservado si `FINAL_FIT_ON_FULL_TRAIN=False`.

## Limitaciones conocidas

Resumen para el informe: [`limitaciones_modelo.md`](limitaciones_modelo.md). La clase `sii = 3` (severa) es minoritaria (~1,2% del train); el recall OOF suele ser bajo aunque el QWK global (~0,41) sea aceptable.

## Notebooks

- `00`–`03`: EDA y modelado baseline
- `04_mejoras_qwk_ensemble.ipynb`: regresión ordinal, ensemble y comparación QWK (lee `experiments/` tras `train-ensemble`)
- `05_modelos_finales_y_comparacion.ipynb`: comparación integrada, tuning y análisis de errores (lee resultados de `experiments/`)
