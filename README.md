## Setup and run

Requires Python 3.10 or newer. From this folder, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_assessment.py
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs_completed.csv --output-dir results/scorer_results
```