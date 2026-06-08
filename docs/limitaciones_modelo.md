# Limitaciones del modelo final

Documento de referencia para el informe y la presentación. Las cifras provienen de `report-oof` (predicciones out-of-fold, sin fuga).

## Desbalance extremo de la clase severa

En el conjunto de entrenamiento, el target ordinal `sii` está muy desbalanceado:

| Clase | Significado (PCIAT) | Participación aprox. en train |
|-------|---------------------|-------------------------------|
| 0 | Ninguno | ~58% |
| 1 | Leve | ~27% |
| 2 | Moderado | ~14% |
| 3 | Severo | ~1,2% |

La clase **3** tiene apenas **34** observaciones en OOF (de 2736 filas). Cualquier métrica por clase en ese nivel tiene **alta varianza** y el modelo recibe pocos ejemplos para aprender el patrón “severo”.

## Desempeño OOF por clase (pipeline final)

Fuente: `experiments/results/per_class_metrics_oof.csv` y `oof_report_summary.json`.

| Clase | Precision OOF | Recall OOF | F1 OOF | Support OOF |
|-------|---------------|------------|--------|-------------|
| 0 | 0,72 | 0,79 | 0,75 | 1594 |
| 1 | 0,37 | 0,32 | 0,34 | 730 |
| 2 | 0,35 | 0,31 | 0,33 | 378 |
| **3** | **0,30** | **0,09** | **0,14** | **34** |

**QWK global OOF:** ~0,41 (métrica principal; coherente con CV 10-fold).

### Interpretación

1. **Clases 0–1** dominan el aprendizaje; el modelo identifica bien la ausencia o levedad de PIU.
2. **Clases 2–3** (moderado/severo) tienen recall bajo; el modelo tiende a **subestimar** la severidad.
3. **Clase 3** es la limitación más crítica: recall ~9% implica que la mayoría de casos severos reales se predicen como 0, 1 o 2.

Esto **no indica un error de implementación**: es el comportamiento esperable con tan pocos positivos de clase 3 y métricas que penalizan errores ordinales lejanos (QWK) frente a F1 macro por clase.

## Decisiones de diseño relacionadas

- **Experimento fusión 2+3:** mejoró F1 macro en holdout pero **redujo QWK**; se mantuvo el target original de 4 clases (`merged_classes_23: false` en `final_model_meta.json`).
- **Métrica de informe:** priorizar **QWK en CV/OOF**, no accuracy ni holdout tras `fit-final` en todo el train.

## Frase sugerida para el informe

> El modelo final alcanza un QWK out-of-fold de aproximadamente 0,41, alineado con la validación cruzada. Sin embargo, el desempeño en la clase severa (`sii = 3`) es limitado (recall OOF ~9%) debido al desbalance extremo (~1,2% del train). Las predicciones son más fiables para distinguir ausencia o levedad de PIU que para detectar casos severos con pocos ejemplos de entrenamiento.

## Cómo actualizar estas cifras

Tras reentrenar o cambiar hiperparámetros:

```bash
python -m src.main fit-final
python -m src.main report-oof
```

Revisar `per_class_metrics_oof.csv` y `oof_report_summary.json` → campo `limitations`.
