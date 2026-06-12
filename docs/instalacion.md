# Instalación y datos

## Requisitos

- Python 3.10–3.14 (recomendado **3.14.2+**, 3.13.11+ o 3.12)
- **Evitar 3.14.1 / 3.13.10** si puedes: bug de `joblib`/`loky` con multiprocessing. El proyecto fuerza `SKLEARN_N_JOBS=1` en esas versiones; con 3.14.2+ puedes usar `SKLEARN_N_JOBS=-1` para CV más rápida.
- Dependencias en `requirements.txt` o `environment.yml` (conda)

**pip / venv:**

```bash
cd proyecto_1_innovacion_tecnologica
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

**conda (opcional):**

```bash
conda env create -f environment.yml
conda activate proyecto-innovacion-tecnologica
```

## Datos (Kaggle)

Coloca los archivos de la competición en:

```
data/raw/train.csv
data/raw/test.csv
```

Origen habitual: carpeta descomprimida del concurso o descarga desde [Kaggle — Child Mind Institute](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data).

## Verificación rápida

```bash
python -m src.main check
```

Debe imprimir las dimensiones del dataset crudo y del dataset listo para modelado (2736 × 46).

## Pipeline de modelado (CLI)

```bash
python -m src.main train           # comparar todos los modelos (CV 10-fold)
python -m src.main train-ensemble  # regresión ordinal + voting (añade filas a los CSV)
python -m src.main tune            # afinar el mejor por QWK (CV 10-fold)
python -m src.main design          # experimento 4 clases vs fusión 2+3
python -m src.main fit-final       # modelo final en todo el train
python -m src.main report-oof    # figuras OOF honestas (~5-15 min)
python -m src.main predict       # submission.csv (tras fit-final)
```

Opcional: `python -m src.main evaluate` solo si `FINAL_FIT_ON_FULL_TRAIN=False`.

Limitaciones del modelo (informe): [`limitaciones_modelo.md`](limitaciones_modelo.md).

Artefactos en `experiments/results/` (CSVs y figuras) y `experiments/checkpoints/` (`final_model.joblib`, `tuned_model.joblib` — versionados en Git).

### Predicción rápida (sin reentrenar)

Si el repo ya incluye los checkpoints:

```bash
pip install -r requirements.txt
# Colocar test.csv en data/raw/
python -m src.main predict   # usa experiments/checkpoints/final_model.joblib
```

Ver [`experiments/checkpoints/README.md`](../experiments/checkpoints/README.md).

Ver [`docs/arquitectura.md`](arquitectura.md).

## Tests

```bash
python -m unittest discover -s tests -v
```

## Notebooks

Abre Jupyter desde la raíz del proyecto para que `src` sea importable:

```bash
jupyter notebook notebooks/
```

Cada notebook tiene celdas iniciales fijas (ver `notebooks/init_notebook.py`):

1. **Init** — `%run init_notebook.py` (imports, rutas, `src.*`, sklearn).
2. **Carga** — según el notebook:
   - `df = load_raw_train()` — notebooks 00, 01, 02 y el notebook completo.
   - `df, df_for_modeling = load_modeling_data()` — notebook 03.
   - Notebook 04 y 05: solo init (leen resultados de `experiments/`).

No repitas imports en celdas posteriores; amplía `init_notebook.py` si hace falta algo global.

Orden sugerido:

1. `00_introduccion_y_carga.ipynb`
2. `01_comprension_calidad_target.ipynb`
3. `02_preparacion_y_eda.ipynb`
4. `03_modelado_evaluacion.ipynb`
5. `04_mejoras_qwk_ensemble.ipynb`
6. `05_modelos_finales_y_comparacion.ipynb`
7. `06_modelo_final_reduced.ipynb` — notebook de entrega (`predict` → `submission.csv`)

Alternativa legacy: `proyecto_innovacion_corregido.ipynb`.
