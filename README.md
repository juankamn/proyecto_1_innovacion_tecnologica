# Child Mind Institute — Problematic Internet Use (Kaggle)

Course project for **Proyecto I de Innovación Tecnológica**, Applied Artificial Intelligence Master, Universidad Icesi, Cali, Colombia.

**Competition:** [Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) (Kaggle)

**Project status:** Active

---

## Team

| Name | GitHub / contact (optional) |
|------|-----------------------------|
| Isabel Cristina Ruiz Buriticá | iris9112@gmail.com |
| Jairo Andrés Valencia | jaanvagu@gmail.com |
| Juan Camilo Macias Navarrete | juankamn@gmail.com |

**Instructor:** Milton Orlando Sarria Paja - mosarria@icesi.edu.co

---

## Project objective

We participate in the Kaggle competition **Child Mind Institute — Problematic Internet Use**. The goal is to build models that help **identify patterns related to problematic internet use (PIU)** in children and adolescents using **de-identified, multi-source data** from the Child Mind Institute’s **Healthy Brain Network (HBN)** study (e.g. questionnaires, clinical and fitness measures, and wearable-derived activity where provided).

A good solution supports **early insight** for researchers and clinicians; final scoring and rules are defined on the [competition page](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) (including the **Evaluation** tab).

---

## Competition context & data

- **Organizers / data:** Child Mind Institute, in collaboration with Kaggle; dataset derives from HBN research protocols.
- **Access:** Data is downloaded from Kaggle after accepting the competition rules. **Do not commit raw competition files** to this repository (size, rules, and team privacy). Use local paths or team-agreed storage outside Git, or document the expected folder layout in `docs/instalacion.md`.
- **Useful links:** [Competition overview](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) · [Data tab](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data)

---

## Methods

- Exploratory analysis and data visualization (notebooks `00`–`02`)  
- Feature engineering and handling of missing / heterogeneous modalities  
- Supervised ML: baselines, classical models (KNN, Ridge ordinal, SVM), boosting (LightGBM, XGBoost)  
- Stratified holdout + **10-fold CV**; primary metric **QWK** (ordinal target `sii`)  
- Hyperparameter tuning, design experiment (merge classes 2+3), error analysis

---

## Technologies

- Python 3.10+  
- Jupyter / notebooks under `notebooks/`  
- Dependencies: [`requirements.txt`](requirements.txt) or [`environment.yml`](environment.yml) (conda)

---

## Repository structure

| Path | Purpose |
|------|---------|
| `docs/` | Installation and architecture ([`docs/README.md`](docs/README.md)) |
| `src/data/` | Load, clean, split (`load.py`, `preprocess.py`, `dataset.py`, `variable_families.py`) |
| `src/models/` | Model pipelines (`baseline`, `classical`, `boosting`, `ordinal`, `registry`) |
| `src/training/` | `train.py`, `train_ensemble.py`, `tune.py`, `run_design_experiments.py`, `fit_final.py` |
| `src/evaluation/` | `metrics.py`, `thresholds.py`, `evaluate.py`, `oof_report.py`, `predict.py` |
| `src/main.py` | CLI (`check`, `train`, `train-ensemble`, `tune`, `design`, `fit-final`, `report-oof`, `predict`) |
| `notebooks/` | EDA and results (`init_notebook.py` for shared setup) |
| `experiments/` | `checkpoints/` (`final_model.joblib`, `tuned_model.joblib`), `results/` (metrics, figures, `submission.csv`) |
| `tests/` | Unit tests |
| `data/raw/` | `train.csv`, `test.csv` (see [`docs/instalacion.md`](docs/instalacion.md)) |

---

## Getting started (contributors)

1. **Clone** this repository.  
2. **Kaggle:** create/join the team account as agreed, accept competition rules, and download the data (browser or [Kaggle API](https://www.kaggle.com/docs/api)).  
3. **Environment:** install dependencies once they are listed in `requirements.txt` or `docs/instalacion.md`.  
4. **Code:** place preprocessing in `src/data/`, training in `src/training/`, and keep exploratory work in `notebooks/`.  
5. **Modeling pipeline:** see [Reproducible workflow](#reproducible-workflow) below.

---

## Reproducible workflow

```bash
pip install -r requirements.txt   # Install dependencies (pandas, sklearn, lightgbm, etc.).
python -m src.main check   # Load train.csv, clean data, verify shape 2736×46.
python -m src.main train   # Train 10 classifiers (dummy, RF, SVM, LightGBM, …). 80/20 split + 10-fold CV. => all_models_holdout.csv, all_models_cv.csv, comparison plot
python -m src.main train-ensemble   # Train 4 ordinal models (3 regressors + vote). Appends rows to same CSVs. => lgbm_regressor, xgb_regressor, etc.
python -m src.main tune   # Pick best by CV and search hyperparameters with RandomizedSearchCV (QWK metric). => tuning_best_params.json, tuned_model.joblib
python -m src.main design   # Test whether merging classes 2+3 improves QWK vs 4 classes. => design_merge_classes_23.csv
python -m src.main fit-final   # Train final model (best model + tune params) and save for delivery. => final_model.joblib, final_model_meta.json
python -m src.main report-oof  # Honest OOF figures for the report
python -m src.main predict     # submission.csv for Kaggle (requires fit-final)
python -m unittest discover -s tests -v
```

---

## Notebooks

| Notebook | Content |
|----------|---------|
| [`notebooks/00_introduccion_y_carga.ipynb`](notebooks/00_introduccion_y_carga.ipynb) | Introduction and data load |
| [`notebooks/01_comprension_calidad_target.ipynb`](notebooks/01_comprension_calidad_target.ipynb) | Data understanding, target, quality |
| [`notebooks/02_preparacion_y_eda.ipynb`](notebooks/02_preparacion_y_eda.ipynb) | Cleaning and EDA |
| [`notebooks/03_modelado_evaluacion.ipynb`](notebooks/03_modelado_evaluacion.ipynb) | Baseline modeling and evaluation |
| [`notebooks/04_mejoras_qwk_ensemble.ipynb`](notebooks/04_mejoras_qwk_ensemble.ipynb) | QWK improvements: ordinal regression, ensemble, preprocessing changes |
| [`notebooks/05_modelos_finales_y_comparacion.ipynb`](notebooks/05_modelos_finales_y_comparacion.ipynb) | Full comparison, tuning, design experiment, error analysis |
| [`notebooks/06_modelo_final_reduced.ipynb`](notebooks/06_modelo_final_reduced.ipynb) | **Definitive submission notebook** — winning pipeline (`lgbm_regressor`) → `submission.csv` |

Local data: `data/raw/train.csv`, `data/raw/test.csv`. See [`docs/instalacion.md`](docs/instalacion.md).

---

## Acknowledgements

Project structure inspired by practices from the [Data Science Working Group](https://github.com/sfbrigade/data-science-wg) (Code for San Francisco).
