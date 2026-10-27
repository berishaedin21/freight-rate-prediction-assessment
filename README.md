# Freight Rate Prediction - worked example

This folder is a reproducible study example for the supplied Machine Learning Engineer assessment. It includes the provided input data and scorer, a forward-in-time validation workflow, final prediction files, the scorer-generated December chart, and a short report.

## Reproduce

Use Python 3.10 or newer. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_assessment.py
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs_completed.csv --output-dir results/scorer_results
```

`run_assessment.py` performs the forward holdout and model comparison, fits the selected approach on all 48,000 labeled rows, and writes:

- `validation_predictions.csv` - 12,000 required validation predictions.
- `december_chart_inputs_completed.csv` - 31 fixed-load December predictions.
- `validation_metrics.json` - holdout split and comparison metrics.

The provided `score.py` validates the two prediction files and creates `results/scorer_results/candidate_december.png`. It does not measure model accuracy; final validation metrics are calculated by Spotter after submission.

## Modeling choices

- Validation is chronological: January-August 2025 for model fitting and September-October 2025 for the holdout.
- The metric used to choose among candidates is MAE; RMSE and WAPE are reported as additional views of error.
- The comparison is between an equipment-specific median rate-per-mile baseline, Ridge regression, and HistGradientBoostingRegressor.
- Pickup, delivery, and equipment are one-hot encoded. Distance, weight, coordinates, market index, quote signal, and calendar features are numeric inputs.
- Numeric missing values are imputed from the training data within each fitted pipeline. Missingness indicators are included. Unknown future categories are ignored safely.
- `load_id` is excluded from model features and used only to align predictions with the submission template.
- The December scenario has no coordinates, market index, or quote signal. Those absent fields are imputed from training data; the resulting chart changes only with date-derived features for the fixed supplied load.

## Result summary

The selected model was HistGradientBoostingRegressor. On the September-October holdout, MAE was $136.94, RMSE was $636.67, WAPE was 5.72%, and mean error (prediction minus actual) was -$49.31. See `validation_metrics.json` and `assessment_report.pdf` for the full comparison and limitations.

## Files

- `data/` contains the original CSV inputs needed to rerun the example.
- `score.py` is the supplied assessment scorer.
- `assessment_instructions.pdf` and `original_readme.md` preserve the supplied task materials.
- `assessment_report.pdf` explains the method, holdout results, data issues, and December chart.
- `loom_walkthrough_script.md` is a draft narration for recording the requested 2-3 minute video.

This is an example for learning and review. It has not been uploaded to GitHub or submitted to the employer. Review the methodology and make your own Loom recording before using any work in a hiring assessment.
