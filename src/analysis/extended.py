"""Approval-gated timing and forecasting specifications."""

from __future__ import annotations

from datetime import timedelta
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.analysis.artifacts import ANALYSIS_ROOT, REPO_ROOT, read_json, relative_path, sha256_file, utc_now_iso, write_json
from src.analysis.descriptive import DESCRIPTIVE_MANIFEST

EXTENDED_ROOT = ANALYSIS_ROOT / "extended"
APPROVAL_RECEIPT = EXTENDED_ROOT / "extended_analysis_approval.json"
TIMING_ARTIFACT = EXTENDED_ROOT / "timing.json"
PREDICTION_ARTIFACT = EXTENDED_ROOT / "prediction.json"
EXTENDED_SCHEMA_VERSION = "extended.v2"
BOOTSTRAP_REPLICATES = 9_999
ESTIMATORS_BY_STAGE = {
    "timing": [
        "audit_window_scaling",
        "fit_segmented_its",
        "placebo_event_diagnostics",
        "pelt_change_points",
        "prewhitened_cross_correlation",
        "distributed_lag_test",
        "granger_predictive_test",
    ],
    "prediction": ["rolling_origin_forecast", "leakage_report"],
}
APPROVAL_BINDINGS = {
    "timing": {
        "stage": "timing",
        "claim_type": "associational_event_timing",
        "events": ["E1", "E2", "E3", "E4", "E5"],
        "outcomes": ["Google Trends daily interest"],
    },
    "prediction": {
        "stage": "prediction",
        "claim_type": "out_of_sample_forecast_comparison",
        "events": ["E1", "E2", "E3", "E4", "E5"],
        "outcomes": ["UK VPN interest", "AU VPN interest"],
    },
}


def rolling_origin_splits(days: list[str] | pd.Series, min_train_days: int = 28, horizons: tuple[int, ...] = (1, 7)) -> list[dict[str, Any]]:
    values = pd.to_datetime(pd.Series(days), errors="raise").sort_values().drop_duplicates().reset_index(drop=True)
    if min_train_days < 1 or not horizons or min(horizons) < 1 or len(values) < min_train_days + max(horizons):
        return []
    splits = []
    for cutoff_index in range(min_train_days, len(values) - max(horizons) + 1):
        cutoff = values.iloc[cutoff_index - 1]
        split = {"cutoff": cutoff.date().isoformat(), "train_end_exclusive": (cutoff + timedelta(days=1)).date().isoformat()}
        for horizon in horizons:
            target = values.iloc[cutoff_index + horizon - 1]
            split[f"target_t_plus_{horizon}"] = target.date().isoformat()
        splits.append(split)
    return splits


def leakage_report(feature_timestamps: pd.Series, cutoff: str) -> dict[str, Any]:
    times = pd.to_datetime(feature_timestamps, errors="coerce")
    cutoff_time = pd.Timestamp(cutoff)
    leaked = times.isna() | (times > cutoff_time)
    return {"rows": int(len(times)), "invalid_timestamp_rows": int(times.isna().sum()), "leaked_rows": int(leaked.sum()), "passed": not bool(leaked.any()), "cutoff": cutoff}


def _validated_time_series(frame: pd.DataFrame, date_column: str, outcome_column: str) -> pd.DataFrame:
    if date_column not in frame or outcome_column not in frame:
        raise ValueError(f"time series needs {date_column} and {outcome_column}")
    values = frame[[date_column, outcome_column]].copy()
    values[date_column] = pd.to_datetime(values[date_column], errors="raise")
    values[outcome_column] = pd.to_numeric(values[outcome_column], errors="raise")
    if values[date_column].duplicated().any() or not np.isfinite(values[outcome_column]).all():
        raise ValueError("time series dates must be unique and outcomes finite")
    return values.sort_values(date_column).reset_index(drop=True)


def _its_design(dates: pd.Series, event_dates: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    time = np.arange(len(dates), dtype=float)
    columns = [np.ones(len(dates)), time]
    names = ["intercept", "time_trend"]
    weekdays = dates.dt.dayofweek.to_numpy()
    for weekday in range(1, 7):
        columns.append((weekdays == weekday).astype(float))
        names.append(f"day_of_week_{weekday}")
    for event_id, event_date in sorted(event_dates.items()):
        point = pd.Timestamp(event_date)
        position = float(np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), np.datetime64(point), side="left"))
        post = (dates >= point).astype(float).to_numpy()
        after = np.maximum(time - position, 0.0)
        columns.extend([post, after])
        names.extend([f"{event_id}_level", f"{event_id}_slope"])
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("segmented ITS design is rank deficient")
    return design, names


def _newey_west_covariance(design: np.ndarray, residuals: np.ndarray, lags: int) -> np.ndarray:
    bread = np.linalg.pinv(design.T @ design)
    meat = design.T @ (residuals[:, None] ** 2 * design)
    for lag in range(1, min(lags, len(residuals) - 1) + 1):
        weight = 1.0 - lag / (lags + 1)
        left = design[lag:] * residuals[lag:, None]
        right = design[:-lag] * residuals[:-lag, None]
        meat += weight * (left.T @ right + right.T @ left)
    return bread @ meat @ bread


def fit_segmented_its(
    frame: pd.DataFrame,
    outcome_column: str,
    event_dates: dict[str, str],
    *,
    date_column: str = "date",
    hac_lags: int = 7,
) -> dict[str, Any]:
    """Fit the predeclared segmented ITS and return HAC-normal uncertainty."""

    if hac_lags < 0:
        raise ValueError("HAC lags must be non-negative")
    values = _validated_time_series(frame, date_column, outcome_column)
    design, names = _its_design(values[date_column], event_dates)
    outcome = values[outcome_column].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
    residuals = outcome - design @ coefficients
    covariance = _newey_west_covariance(design, residuals, hac_lags)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    rows = {}
    for name, estimate, standard_error in zip(names, coefficients, standard_errors):
        z_score = float(estimate / standard_error) if standard_error else (0.0 if estimate == 0 else float("inf"))
        rows[name] = {
            "estimate": float(estimate),
            "standard_error": float(standard_error),
            "lower_95": float(estimate - 1.96 * standard_error),
            "upper_95": float(estimate + 1.96 * standard_error),
            "z": z_score,
            "p_value_two_sided": float(2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / np.sqrt(2))))) if np.isfinite(z_score) else 0.0,
        }
    return {
        "status": "estimated",
        "date_column": date_column,
        "outcome_column": outcome_column,
        "event_dates": event_dates,
        "rows": int(len(values)),
        "hac_lags": int(hac_lags),
        "r_squared": float(1 - (residuals @ residuals) / np.sum((outcome - outcome.mean()) ** 2)) if np.var(outcome) else 0.0,
        "coefficients": rows,
    }


def audit_window_scaling(
    frame: pd.DataFrame,
    *,
    window_column: str = "window",
    date_column: str = "date",
    value_column: str = "interest",
    scaling_column: str = "scaling_factor",
    minimum_overlap: int = 3,
) -> pd.DataFrame:
    """Audit adjacent-window overlap before accepting a stitched series."""

    required = {window_column, date_column, value_column, scaling_column}
    if not required <= set(frame.columns):
        raise ValueError(f"scaling audit missing columns: {sorted(required - set(frame.columns))}")
    values = frame.copy()
    values[date_column] = pd.to_datetime(values[date_column], errors="raise")
    values[value_column] = pd.to_numeric(values[value_column], errors="raise")
    values[scaling_column] = pd.to_numeric(values[scaling_column], errors="raise")
    if not np.isfinite(values[[value_column, scaling_column]].to_numpy(dtype=float)).all() or values[scaling_column].le(0).any():
        raise ValueError("scaling values must be finite and positive")
    windows = sorted(values[window_column].astype(str).unique())
    rows = []
    for previous, current in zip(windows, windows[1:]):
        left = values[values[window_column].astype(str).eq(previous)].set_index(date_column)[value_column]
        right = values[values[window_column].astype(str).eq(current)].set_index(date_column)[value_column]
        overlap = pd.concat([left.rename("previous"), right.rename("current")], axis=1, join="inner").dropna()
        ratios = overlap["previous"] / overlap["current"].replace(0, np.nan)
        factor = values.loc[values[window_column].astype(str).eq(current), scaling_column]
        if factor.nunique() != 1:
            raise ValueError(f"scaling factor must be constant within window: {current}")
        rows.append(
            {
                "previous_window": previous,
                "current_window": current,
                "overlap_rows": int(len(overlap)),
                "median_raw_ratio": float(ratios.median()) if ratios.notna().any() else None,
                "declared_scaling_factor": float(factor.iloc[0]) if not factor.empty else None,
                "minimum_overlap": minimum_overlap,
                "passed": bool(len(overlap) >= minimum_overlap and ratios.notna().any()),
            }
        )
    return pd.DataFrame(rows)


def placebo_event_diagnostics(
    frame: pd.DataFrame,
    outcome_column: str,
    event_date: str,
    placebo_dates: list[str],
    *,
    date_column: str = "date",
) -> dict[str, Any]:
    """Fit the same segmented design at false dates as a timing falsification check."""

    values = _validated_time_series(frame, date_column, outcome_column)
    pre = values[values[date_column] < pd.Timestamp(event_date)]
    pretrend = None
    if len(pre) >= 2:
        pretrend = float(np.polyfit(np.arange(len(pre), dtype=float), pre[outcome_column].to_numpy(dtype=float), 1)[0])
    rows = []
    for index, placebo_date in enumerate(placebo_dates):
        result = fit_segmented_its(values, outcome_column, {f"placebo_{index}": placebo_date}, date_column=date_column)
        coefficient = result["coefficients"][f"placebo_{index}_level"]
        rows.append({"placebo_date": placebo_date, **coefficient})
    return {"event_date": event_date, "pretrend_slope": pretrend, "placebo_results": rows}


def pelt_change_points(values: pd.Series | np.ndarray, *, penalty: float, minimum_segment: int = 7) -> dict[str, Any]:
    """Find penalized mean-change points with the PELT candidate-pruning rule."""

    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or minimum_segment < 1 or len(series) < 2 * minimum_segment or not np.isfinite(series).all() or penalty <= 0:
        raise ValueError("change-point input is invalid or too short")
    prefix = np.r_[0.0, np.cumsum(series)]
    squared = np.r_[0.0, np.cumsum(series * series)]

    def cost(start: int, end: int) -> float:
        count = end - start
        total = prefix[end] - prefix[start]
        return float(squared[end] - squared[start] - total * total / count)

    scores = np.full(len(series) + 1, np.inf)
    previous = np.full(len(series) + 1, -1, dtype=int)
    scores[0] = -penalty
    candidates = [0]
    for end in range(minimum_segment, len(series) + 1):
        valid = [start for start in candidates if end - start >= minimum_segment]
        if not valid:
            continue
        start = min(valid, key=lambda value: scores[value] + cost(value, end) + penalty)
        scores[end] = scores[start] + cost(start, end) + penalty
        previous[end] = start
        # ponytail: scalar PELT is O(n^2) in the worst case; keep the exact
        # method local and move to a compiled implementation only if profiling
        # shows this sensitivity is material.
        candidates = [value for value in candidates if end - value < minimum_segment or scores[value] + cost(value, end) <= scores[end]]
        candidates.append(end)
    points = []
    end = len(series)
    while previous[end] > 0:
        points.append(int(previous[end]))
        end = previous[end]
    return {"method": "pelt_mean_change_point", "penalty": penalty, "minimum_segment": minimum_segment, "change_points": sorted(points), "objective": float(scores[-1])}


def _prewhiten_pair(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    first = np.asarray(left, dtype=float)
    second = np.asarray(right, dtype=float)
    if first.ndim != 1 or second.ndim != 1 or len(first) != len(second) or len(first) < 5 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("cross-correlation inputs must be equal finite vectors")
    first, second = np.diff(first), np.diff(second)
    phi = float(np.linalg.lstsq(first[:-1, None], first[1:], rcond=None)[0][0]) if len(first) > 1 else 0.0
    return first[1:] - phi * first[:-1], second[1:] - phi * second[:-1], phi


def prewhitened_cross_correlation(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray, *, maximum_lag: int = 14) -> pd.DataFrame:
    """Correlate differenced series after removing a shared AR(1) filter."""

    if maximum_lag < 0:
        raise ValueError("maximum_lag must be non-negative")
    first, second, phi = _prewhiten_pair(left, right)
    rows = []
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            left_values, right_values = first[:lag], second[-lag:]
        elif lag > 0:
            left_values, right_values = first[lag:], second[:-lag]
        else:
            left_values, right_values = first, second
        correlation = np.corrcoef(left_values, right_values)[0, 1] if len(left_values) >= 3 else np.nan
        rows.append({"lag": lag, "correlation": float(correlation) if np.isfinite(correlation) else None, "rows": int(len(left_values)), "ar1_phi": phi})
    return pd.DataFrame(rows)


def distributed_lag_test(target: pd.Series | np.ndarray, predictor: pd.Series | np.ndarray, *, maximum_lag: int = 7) -> dict[str, Any]:
    """Compare target-lag-only and target-plus-predictor-lag linear forecasts."""

    y = np.asarray(target, dtype=float)
    x = np.asarray(predictor, dtype=float)
    if maximum_lag < 1 or y.ndim != 1 or x.ndim != 1 or len(y) != len(x) or len(y) <= maximum_lag + 3 or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("distributed-lag inputs are invalid or too short")
    rows = []
    for lag in range(1, maximum_lag + 1):
        target_lags = np.column_stack([y[lag - offset : len(y) - offset] for offset in range(1, lag + 1)])
        predictor_lags = np.column_stack([x[lag - offset : len(x) - offset] for offset in range(1, lag + 1)])
        response = y[lag:]
        restricted = np.column_stack([np.ones(len(response)), target_lags])
        unrestricted = np.column_stack([restricted, predictor_lags])
        restricted_residuals = response - restricted @ np.linalg.lstsq(restricted, response, rcond=None)[0]
        unrestricted_residuals = response - unrestricted @ np.linalg.lstsq(unrestricted, response, rcond=None)[0]
        numerator = (restricted_residuals @ restricted_residuals - unrestricted_residuals @ unrestricted_residuals) / lag
        denominator = (unrestricted_residuals @ unrestricted_residuals) / max(len(response) - unrestricted.shape[1], 1)
        statistic = max(float(numerator / denominator), 0.0) if denominator else 0.0
        rows.append({"lag": lag, "f_statistic": statistic, "p_value": min(1.0, float(f_distribution.sf(statistic, lag, max(len(response) - unrestricted.shape[1], 1)))), "rows": int(len(response))})
    return {"method": "distributed_lag_ols", "results": rows}


def granger_predictive_test(target: pd.Series | np.ndarray, predictor: pd.Series | np.ndarray, *, maximum_lag: int = 7) -> dict[str, Any]:
    """Run the named predictive lag test on differenced inputs after pre-whitening."""

    correlation = prewhitened_cross_correlation(target, predictor, maximum_lag=maximum_lag)
    whitened_target, whitened_predictor, _ = _prewhiten_pair(target, predictor)
    result = distributed_lag_test(whitened_target, whitened_predictor, maximum_lag=maximum_lag)
    return {**result, "method": "prewhitened_differenced_granger_predictive_test", "cross_correlation": correlation.to_dict(orient="records")}


def _forecast_features(values: pd.DataFrame, target_column: str, feature_columns: tuple[str, ...]) -> pd.DataFrame:
    result = values.copy()
    result["lag_1"] = result[target_column].shift(1)
    result["lag_7"] = result[target_column].shift(7)
    columns = ["lag_1", "lag_7", *feature_columns]
    result[columns] = result[columns].apply(pd.to_numeric, errors="raise")
    return result


def _model_predictions(
    model_name: str,
    train: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
    date_column: str,
) -> float:
    if model_name == "seasonal_naive":
        seasonal_date = target.index[0] - timedelta(days=7)
        historical = train.set_index(date_column)
        return float(historical[target_column].get(seasonal_date, train[target_column].iloc[-1]))
    columns = ["lag_1", "lag_7", *feature_columns]
    usable = train.dropna(subset=columns + [target_column])
    if usable.empty or target[columns].isna().any(axis=None):
        raise ValueError(f"insufficient lagged features for {model_name}")
    if model_name == "autoregressive_lag_only":
        model = LinearRegression().fit(usable[["lag_1", "lag_7"]], usable[target_column])
        return float(model.predict(target[["lag_1", "lag_7"]])[0])
    if model_name == "elastic_net":
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=20260915).fit(usable[columns], usable[target_column])
    elif model_name == "histogram_gradient_boosting":
        model = HistGradientBoostingRegressor(random_state=20260915, max_iter=100).fit(usable[columns], usable[target_column])
    else:
        raise ValueError(f"unknown forecast model: {model_name}")
    return float(model.predict(target[columns])[0])


def rolling_origin_forecast(
    frame: pd.DataFrame,
    target_column: str,
    *,
    date_column: str = "date",
    feature_columns: tuple[str, ...] = (),
    feature_time_column: str | None = None,
    min_train_days: int = 28,
    horizons: tuple[int, ...] = (1, 7),
) -> dict[str, Any]:
    """Evaluate predeclared forecasts with rolling-origin, cutoff-safe features."""

    values = frame.copy()
    _validated_time_series(values, date_column, target_column)
    values[date_column] = pd.to_datetime(values[date_column], errors="raise")
    values[target_column] = pd.to_numeric(values[target_column], errors="raise")
    missing = sorted(set(feature_columns) - set(values.columns))
    if missing:
        raise ValueError(f"forecast features missing: {missing}")
    if feature_time_column is not None and feature_time_column not in values:
        raise ValueError(f"forecast feature timestamp missing: {feature_time_column}")
    if feature_time_column is not None:
        values[feature_time_column] = pd.to_datetime(values[feature_time_column], errors="raise")
        if (values[feature_time_column] > values[date_column]).any():
            raise ValueError("forecast feature timestamp is after its observation date")
    values = values.sort_values(date_column).reset_index(drop=True)
    values = _forecast_features(values, target_column, feature_columns)
    dates = values[date_column]
    by_date = values.set_index(date_column)
    models = ("seasonal_naive", "autoregressive_lag_only", "elastic_net", "histogram_gradient_boosting")
    rows = []
    for split in rolling_origin_splits(dates, min_train_days=min_train_days, horizons=horizons):
        cutoff = split["cutoff"]
        train = values[dates.dt.date.astype(str) <= cutoff]
        for horizon in horizons:
            target_date = split[f"target_t_plus_{horizon}"]
            if pd.Timestamp(target_date) not in by_date.index:
                continue
            target = by_date.loc[[pd.Timestamp(target_date)]]
            if feature_time_column is not None:
                if (train[feature_time_column] > pd.Timestamp(cutoff)).any():
                    raise ValueError(f"forecast training feature leakage at {cutoff}")
                safe = leakage_report(target[feature_time_column], cutoff)
                if not safe["passed"]:
                    raise ValueError(f"forecast feature leakage at {cutoff}")
            for model_name in models:
                prediction = _model_predictions(model_name, train, target, feature_columns, target_column, date_column)
                rows.append({"cutoff": cutoff, "target_date": target_date, "horizon": horizon, "model": model_name, "actual": float(target[target_column].iloc[0]), "predicted": prediction})
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        return {"status": "no_rolling_origin_splits", "predictions": [], "metrics": {}}
    metrics = {}
    for (model_name, horizon), group in predictions.groupby(["model", "horizon"], sort=True):
        baseline = predictions[(predictions["model"] == "autoregressive_lag_only") & predictions["horizon"].eq(horizon)]
        paired = group.merge(
            baseline[["cutoff", "target_date", "actual", "predicted"]].rename(columns={"actual": "baseline_actual", "predicted": "baseline_predicted"}),
            on=["cutoff", "target_date"],
            how="inner",
        )
        improvement_interval = None
        if model_name != "autoregressive_lag_only" and not paired.empty:
            errors = np.abs(paired["actual"].to_numpy(dtype=float) - paired["predicted"].to_numpy(dtype=float))
            baseline_errors = np.abs(paired["baseline_actual"].to_numpy(dtype=float) - paired["baseline_predicted"].to_numpy(dtype=float))
            if np.any(baseline_errors):
                rng = np.random.default_rng(20260915 + int(horizon))
                draws = []
                for _ in range(BOOTSTRAP_REPLICATES):
                    indexes = rng.integers(len(errors), size=len(errors))
                    draws.append(1.0 - errors[indexes].mean() / baseline_errors[indexes].mean() if baseline_errors[indexes].mean() else 0.0)
                improvement_interval = {"lower_95": float(np.quantile(draws, 0.025)), "median": float(np.quantile(draws, 0.5)), "upper_95": float(np.quantile(draws, 0.975)), "replicates": BOOTSTRAP_REPLICATES}
        metrics[f"{model_name}:t+{horizon}"] = {
            "mae": float(mean_absolute_error(group["actual"], group["predicted"])),
            "rmse": float(np.sqrt(mean_squared_error(group["actual"], group["predicted"]))),
            "rows": int(len(group)),
            "improvement_over_lag_only_mae": None if model_name == "autoregressive_lag_only" else float(1 - mean_absolute_error(group["actual"], group["predicted"]) / mean_absolute_error(baseline["actual"], baseline["predicted"])) if not baseline.empty and mean_absolute_error(baseline["actual"], baseline["predicted"]) else None,
            "improvement_over_lag_only_mae_interval": improvement_interval,
        }
    return {"status": "evaluated", "models": list(models), "horizons": list(horizons), "predictions": predictions.to_dict(orient="records"), "metrics": metrics, "leakage_checked": feature_time_column is not None}


def _source_checksums() -> dict[str, str | None]:
    paths = {
        "descriptive_manifest": DESCRIPTIVE_MANIFEST,
        "google_trends_daily": REPO_ROOT / "data" / "processed" / "google_trends" / "google_trends_daily.csv",
        "google_trends_scaling": REPO_ROOT / "data" / "processed" / "google_trends" / "google_trends_scaling_factors.csv",
    }
    return {name: sha256_file(path) if path.exists() else None for name, path in paths.items()}


def _approval_status(stage: str) -> tuple[str, str | None]:
    if not APPROVAL_RECEIPT.exists():
        return "approval-required", None
    receipt = read_json(APPROVAL_RECEIPT)
    binding = receipt.get("binding")
    if (
        receipt.get("approved_for_extended_analysis") is not True
        or not receipt.get("approved_by")
        or not receipt.get("approved_at_utc")
        or binding != APPROVAL_BINDINGS[stage]
    ):
        return "approval-receipt-invalid", sha256_file(APPROVAL_RECEIPT)
    return "approved", sha256_file(APPROVAL_RECEIPT)


def _payload(stage: str) -> dict[str, Any]:
    status, receipt_checksum = _approval_status(stage)
    payload: dict[str, Any] = {
        "schema_version": EXTENDED_SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "stage": stage,
        "approval_receipt_path": relative_path(APPROVAL_RECEIPT),
        "approval_receipt_sha256": receipt_checksum,
        "approval_binding": APPROVAL_BINDINGS[stage],
        "source_checksums": _source_checksums(),
        "assessed_output_written": False,
        "status": status,
        "estimators_implemented": ESTIMATORS_BY_STAGE[stage],
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    if stage == "timing":
        payload.update(
            {
                "design": {
                    "series": "frozen Google Trends daily series stitched only from existing files",
                    "model": "segmented interrupted time-series with day-of-week terms, trend terms and HAC uncertainty",
                    "events": ["E2", "E3", "E4", "E5"],
                    "contextual_negative_control": "E1",
                    "comparators": ["Ireland", "New Zealand"],
                    "change_point_sensitivity": "PELT or Bayesian change-point; corroboration only",
                    "lag_tests": "pre-whitened cross-correlation and distributed-lag/Granger predictive tests",
                },
                "required_receipt": "data/analysis/extended/extended_analysis_approval.json",
            }
        )
    else:
        payload.update(
            {
                "design": {
                    "targets": ["UK VPN interest t+1", "UK VPN interest t+7", "AU VPN interest t+1", "AU VPN interest t+7"],
                    "models": ["seasonal_naive", "autoregressive_lag_only", "elastic_net", "histogram_gradient_boosting"],
                    "evaluation": "rolling-origin MAE/RMSE with uncertainty in improvement over lag-only baseline",
                    "features": "timestamped lagged target, calendar, event phase, validated discourse and pre-cutoff network summaries",
                    "leakage_gate": "every feature timestamp must be at or before the forecast cutoff",
                },
                "required_receipt": "data/analysis/extended/extended_analysis_approval.json",
            }
        )
    if status != "approved":
        payload["blocking_reason"] = "Extended analysis is outside the frozen core until a current lecturer/team approval receipt is present; no assessed result is a zero or null finding."
    else:
        payload["status"] = "implementation-required"
        payload["blocking_reason"] = "Approval is present, but the frozen snapshot has no executed extended-model result; implementation must be reviewed before activation."
    return payload


def check_extended(stage: str) -> dict[str, Any]:
    if stage not in {"timing", "prediction"}:
        raise ValueError(f"unknown extended stage: {stage}")
    path = TIMING_ARTIFACT if stage == "timing" else PREDICTION_ARTIFACT
    if not path.exists():
        raise FileNotFoundError(f"extended artifact is missing: {path}; run the explicit extended {stage} build first")
    actual = read_json(path)
    if actual.get("schema_version") != EXTENDED_SCHEMA_VERSION:
        raise ValueError(f"extended artifact schema is stale: {path}; run the explicit extended {stage} build")
    if actual.get("source_checksums") != _source_checksums():
        raise ValueError(f"extended source inputs changed: {path}; run the explicit extended {stage} build")
    if actual.get("estimators_implemented") != ESTIMATORS_BY_STAGE[stage] or actual.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
        raise ValueError(f"extended artifact contract is stale: {path}; run the explicit extended {stage} build")
    if actual.get("approval_binding") != APPROVAL_BINDINGS[stage] or actual.get("approval_receipt_sha256") != (_approval_status(stage)[1]):
        raise ValueError(f"extended approval state changed: {path}; run the explicit extended {stage} build")
    return {"status": actual["status"], "artifact": relative_path(path), "assessed_output_written": actual["assessed_output_written"]}


def build_extended(stage: str) -> dict[str, Any]:
    if stage not in {"timing", "prediction"}:
        raise ValueError(f"unknown extended stage: {stage}")
    path = TIMING_ARTIFACT if stage == "timing" else PREDICTION_ARTIFACT
    payload = _payload(stage)
    EXTENDED_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)
    return {"status": payload["status"], "artifact": relative_path(path), "assessed_output_written": False}
