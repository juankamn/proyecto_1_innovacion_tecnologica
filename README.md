# Child Mind Institute — Problematic Internet Use (Kaggle)

Course project for **Proyecto 1 de Innovación Tecnológica**, Applied Artificial Intelligence Master, Universidad Icesi, Cali, Colombia.

**Competition:** [Child Mind Institute — Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) (Kaggle)

**Project status:** Active

---

## Team

| Name | GitHub / contact (optional) |
|------|-----------------------------|
| Isabel Cristina Ruiz Buriticá | _TBD_ |
| Jairo Andrés Valencia | _TBD_ |
| Juan Camilo Macias Navarrete | _TBD_ |

**Instructor:** _TBD_

_Add team roles (e.g. coordination) here if you use them._

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

## Methods (planned)

- Exploratory analysis and data visualization  
- Feature engineering and handling of missing / heterogeneous modalities  
- Supervised machine learning (models and validation strategy aligned with the competition)  
- Error analysis and iteration on the leaderboard metric (see Kaggle **Evaluation**)

---

## Technologies (planned)

- Python 3.x  
- Jupyter / notebooks under `notebooks/`  
- `pandas`, `numpy`; modeling stack to be pinned in `requirements.txt` / `environment.yml` (e.g. scikit-learn, gradient boosting libraries, etc.)

---

## Repository layout (high level)

| Path | Purpose |
|------|---------|
| `src/data/` | Load and preprocess data (e.g. `preprocess.py`) |
| `src/models/` | Model definitions |
| `src/training/` | Training scripts |
| `src/evaluation/` | Evaluation helpers |
| `src/utils/` | Shared utilities |
| `src/main.py` | Entry point (when wired) |
| `notebooks/` | Experiments and EDA |
| `experiments/` | Logs, checkpoints, results (local; usually gitignored) |
| `docs/` | Extended docs (`docs/instalacion.md`, etc.) |

---

## Getting started (contributors)

1. **Clone** this repository.  
2. **Kaggle:** create/join the team account as agreed, accept competition rules, and download the data (browser or [Kaggle API](https://www.kaggle.com/docs/api)).  
3. **Environment:** install dependencies once they are listed in `requirements.txt` or `docs/instalacion.md`.  
4. **Code:** place preprocessing in `src/data/`, training in `src/training/`, and keep exploratory work in `notebooks/`.  
5. **Submissions:** generated locally; follow Kaggle’s submission format and deadlines.

---

## Featured notebooks / deliverables

- _To add:_ links to key notebooks, reports, or final submission notes.

---

## Acknowledgements

Project structure inspired by practices from the [Data Science Working Group](https://github.com/sfbrigade/data-science-wg) (Code for San Francisco).
