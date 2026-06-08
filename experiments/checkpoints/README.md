# Model checkpoints (versioned)

| File | Purpose | Used by |
|------|---------|---------|
| `final_model.joblib` | **Production model** — trained on full labeled train | `python -m src.main predict`, notebook `06_modelo_final_reduced` |
| `tuned_model.joblib` | **Tuning artifact** — best pipeline from `RandomizedSearchCV` on the 80/20 train split | Inspection / reproducibility of the `tune` step; **not** used for Kaggle submission |

To regenerate locally:

```bash
python -m src.main tune        # tuned_model.joblib + tuning_best_params.json
python -m src.main fit-final   # final_model.joblib + final_model_meta.json
```

Load with the same dependency versions as `requirements.txt` / `environment.yml` (Python, scikit-learn, LightGBM).
