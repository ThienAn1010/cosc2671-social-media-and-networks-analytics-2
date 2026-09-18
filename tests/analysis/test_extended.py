from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.extended import (
    distributed_lag_test,
    fit_segmented_its,
    granger_predictive_test,
    leakage_report,
    pelt_change_points,
    placebo_event_diagnostics,
    rolling_origin_forecast,
    rolling_origin_splits,
)


def test_rolling_origin_splits_keep_targets_after_cutoff():
    days = pd.date_range("2025-01-01", periods=40, freq="D")
    splits = rolling_origin_splits(days, min_train_days=28)
    assert splits
    assert splits[0]["cutoff"] < splits[0]["target_t_plus_1"]


def test_leakage_report_fails_for_future_feature():
    report = leakage_report(pd.Series(["2025-01-01", "2025-01-03"]), "2025-01-02")
    assert report["passed"] is False
    assert report["leaked_rows"] == 1


def test_segmented_its_returns_hac_coefficients():
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    frame = pd.DataFrame({"date": dates, "outcome": np.linspace(0, 1, len(dates))})

    result = fit_segmented_its(frame, "outcome", {"E1": "2025-01-21", "E2": "2025-02-10"})

    assert result["status"] == "estimated"
    assert "E1_level" in result["coefficients"]
    assert result["coefficients"]["time_trend"]["standard_error"] >= 0


def test_extended_falsification_and_lag_estimators_return_declared_outputs():
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    frame = pd.DataFrame({"date": dates, "outcome": np.r_[np.zeros(40), np.ones(40)]})

    placebo = placebo_event_diagnostics(frame, "outcome", "2025-02-10", ["2025-01-21"])
    change_points = pelt_change_points(frame["outcome"].to_numpy(), penalty=2.0, minimum_segment=7)
    lags = distributed_lag_test(np.sin(np.arange(80) / 5), np.cos(np.arange(80) / 5), maximum_lag=3)
    granger = granger_predictive_test(np.sin(np.arange(80) / 5), np.cos(np.arange(80) / 5), maximum_lag=3)

    assert len(placebo["placebo_results"]) == 1
    assert change_points["method"] == "pelt_mean_change_point"
    assert len(lags["results"]) == 3
    assert granger["method"].startswith("prewhitened_")


def test_rolling_origin_forecast_evaluates_baselines_and_models_without_leakage():
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "outcome": np.sin(np.arange(len(dates)) / 5) + np.arange(len(dates)) / 100,
            "feature": np.arange(len(dates), dtype=float),
            "feature_time": dates - pd.Timedelta(days=7),
        }
    )

    result = rolling_origin_forecast(frame, "outcome", feature_columns=("feature",), feature_time_column="feature_time", min_train_days=28)

    assert result["status"] == "evaluated"
    assert set(result["models"]) == {"seasonal_naive", "autoregressive_lag_only", "elastic_net", "histogram_gradient_boosting"}
    assert result["leakage_checked"] is True
    assert result["metrics"]["elastic_net:t+1"]["improvement_over_lag_only_mae_interval"]["replicates"] == 9999
