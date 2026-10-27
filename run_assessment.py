from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CATEGORICAL = ["pickup", "delivery", "equipment"]
NUMERIC = [
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "distance", "weight", "market_index", "quote_signal",
    "year", "month", "dayofyear", "dayofweek", "is_weekend",
    "month_sin", "month_cos", "date_ordinal",
]


def features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create only features available at quote time; load_id is deliberately excluded."""
    out = frame.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid date values found")
    out["year"] = dates.dt.year
    out["month"] = dates.dt.month
    out["dayofyear"] = dates.dt.dayofyear
    out["dayofweek"] = dates.dt.dayofweek
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
    angle = 2 * np.pi * dates.dt.dayofyear / 365.25
    out["month_sin"] = np.sin(angle)
    out["month_cos"] = np.cos(angle)
    out["date_ordinal"] = dates.map(pd.Timestamp.toordinal).astype(float)
    # The chart scenario intentionally contains fewer fields than the main validation file.
    # Missing columns are treated as unavailable and imputed by the fitted training pipeline.
    for column in CATEGORICAL + NUMERIC:
        if column not in out.columns:
            out[column] = np.nan
    return out[CATEGORICAL + NUMERIC]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "wape_percent": float(100 * np.sum(np.abs(error)) / np.sum(np.abs(y_true))),
        "mean_error_prediction_minus_actual": float(np.mean(error)),
    }


def make_models() -> dict[str, Pipeline]:
    linear_preprocess = ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), CATEGORICAL),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler(with_mean=False)),
            ]), NUMERIC),
        ],
        remainder="drop",
    )
    tree_preprocess = ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL),
            ("numeric", SimpleImputer(strategy="median", add_indicator=True), NUMERIC),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return {
        "ridge": Pipeline([
            ("preprocess", linear_preprocess),
            ("model", Ridge(alpha=30.0, solver="lsqr")),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocess", tree_preprocess),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_iter=220,
                max_leaf_nodes=31,
                min_samples_leaf=50,
                l2_regularization=5.0,
                early_stopping=False,
                random_state=42,
            )),
        ]),
    }


def equipment_rate_per_mile(train: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    ratios = train.assign(rate_per_mile=train.posted_rate / train.distance).groupby("equipment").rate_per_mile.median()
    overall = float((train.posted_rate / train.distance).median())
    return frame.distance.to_numpy(dtype=float) * frame.equipment.map(ratios).fillna(overall).to_numpy(dtype=float)


def main() -> None:
    train = pd.read_csv(DATA / "train-test.csv")
    validation = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation-predictions-template.csv", dtype={"load_id": str})
    december = pd.read_csv(DATA / "december-chart-inputs.csv")
    train["date"] = pd.to_datetime(train["date"])
    validation["date"] = pd.to_datetime(validation["date"])
    december["date"] = pd.to_datetime(december["date"])

    # Forward-in-time holdout: fit on Jan-Aug, evaluate once on Sep-Oct.
    cutoff = pd.Timestamp("2025-09-01")
    dev_train = train.loc[train.date < cutoff].copy()
    dev_valid = train.loc[train.date >= cutoff].copy()
    x_train, y_train = features(dev_train), dev_train.posted_rate.to_numpy(dtype=float)
    x_valid, y_valid = features(dev_valid), dev_valid.posted_rate.to_numpy(dtype=float)

    comparison: dict[str, dict[str, float]] = {}
    comparison["equipment_median_rate_per_mile_baseline"] = metrics(
        y_valid, equipment_rate_per_mile(dev_train, dev_valid)
    )
    validation_models: dict[str, Pipeline] = {}
    for name, model in make_models().items():
        model.fit(x_train, y_train)
        validation_models[name] = model
        prediction = np.maximum(model.predict(x_valid), 1.0)
        comparison[name] = metrics(y_valid, prediction)

    selected_name = min(comparison, key=lambda name: comparison[name]["mae"])
    final_train = train.copy()
    if selected_name == "equipment_median_rate_per_mile_baseline":
        final_validation_prediction = equipment_rate_per_mile(final_train, validation)
        final_december_prediction = equipment_rate_per_mile(final_train, december)
    else:
        final_model = make_models()[selected_name]
        final_model.fit(features(final_train), final_train.posted_rate.to_numpy(dtype=float))
        final_validation_prediction = final_model.predict(features(validation))
        final_december_prediction = final_model.predict(features(december))

    final_validation_prediction = np.maximum(final_validation_prediction, 1.0)
    final_december_prediction = np.maximum(final_december_prediction, 1.0)
    if len(template) != len(validation) or set(template.load_id) != set(validation.load_id.astype(str)):
        raise ValueError("Prediction template IDs do not exactly match validation IDs")
    rate_by_id = pd.Series(final_validation_prediction, index=validation.load_id.astype(str))
    submission = template[["load_id"]].copy()
    submission["predicted_rate"] = submission.load_id.map(rate_by_id)
    if submission.predicted_rate.isna().any():
        raise ValueError("Some template IDs did not receive a prediction")
    submission.to_csv(ROOT / "validation_predictions.csv", index=False, float_format="%.2f")

    december_output = december.copy()
    december_output["date"] = december_output.date.dt.strftime("%Y-%m-%d")
    december_output["predicted_rate"] = final_december_prediction
    december_output.to_csv(ROOT / "december_chart_inputs_completed.csv", index=False, float_format="%.2f")

    summary = {
        "validation_strategy": "Forward holdout: train on 2025-01-01 through 2025-08-31; validate on 2025-09-01 through 2025-10-31",
        "development_rows": int(len(train)),
        "forward_training_rows": int(len(dev_train)),
        "forward_validation_rows": int(len(dev_valid)),
        "missing_training_weight": int(train.weight.isna().sum()),
        "missing_training_market_index": int(train.market_index.isna().sum()),
        "missing_validation_weight": int(validation.weight.isna().sum()),
        "missing_validation_market_index": int(validation.market_index.isna().sum()),
        "selected_model_by_forward_holdout_mae": selected_name,
        "forward_holdout_metrics": comparison,
        "final_validation_rows": int(len(submission)),
        "final_december_rows": int(len(december_output)),
        "features_exclude_load_id": True,
        "notes": [
            "Numeric missing values are imputed using medians learned from each training fold; missingness indicators are included.",
            "Categorical pickup, delivery, and equipment fields are one-hot encoded; unseen future categories are ignored safely.",
            "The supplied score.py checks output structure and creates the December chart; it does not calculate prediction accuracy.",
        ],
    }
    (ROOT / "validation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Created {ROOT / 'validation_predictions.csv'}")
    print(f"Created {ROOT / 'december_chart_inputs_completed.csv'}")


if __name__ == "__main__":
    main()


