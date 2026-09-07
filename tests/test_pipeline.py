from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    Config, build_synthetic_csv, classify_event, load_energy_csv,
    make_features, make_windows, minmax01, robust_normal_mask,
    run_pipeline,
)


def test_load_clean_and_resample(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({
        "Datetime": ["2025-01-01 00:00", "2025-01-01 00:00", "2025-01-01 02:00"],
        "PJME_MW": [100, 120, 140],
    }).to_csv(p, index=False)
    df = load_energy_csv(p, "PJME_MW")
    assert len(df) == 3
    assert df["Energy_MW"].isna().sum() == 0
    assert df.iloc[0].Energy_MW == 110
    assert df.iloc[1].Energy_MW == 125


def test_features_and_windows():
    dt = pd.date_range("2025-01-01", periods=50, freq="h")
    df = pd.DataFrame({"Datetime": dt, "Energy_MW": np.arange(50, dtype=float)})
    f = make_features(df)
    for c in ["sin_hour", "cos_hour", "sin_week", "cos_week", "rolling_mean_24", "delta_1h"]:
        assert c in f
    w, s = make_windows(np.arange(50.), 10, 5)
    assert w.shape == (9, 10)
    assert np.array_equal(s[:3], [0, 5, 10])


def test_invalid_windows():
    with pytest.raises(ValueError):
        make_windows(np.arange(5.), 10, 1)
    with pytest.raises(ValueError):
        make_windows(np.arange(20.).reshape(4, 5), 5, 1)


def test_normal_mask_and_minmax():
    x = np.array([10, 10, 11, 9, 10, 100], float)
    mask = robust_normal_mask(x, 3)
    assert mask[-1] is False or bool(mask[-1]) is False
    z = minmax01(np.array([2., 4., 6.]))
    assert np.allclose(z, [0, .5, 1])


def test_event_classification():
    normal = np.ones(24) * 100
    spike = normal.copy(); spike[12] = 1000
    drop = normal.copy(); drop[12] = -800
    sustained = normal.copy(); sustained[5:20] = 400
    assert classify_event(normal, 0.1, 0.5) == "Normal"
    assert classify_event(spike, 0.9, 0.5) == "Local spike"
    assert classify_event(drop, 0.9, 0.5) == "Local drop"
    assert classify_event(sustained, 0.9, 0.5) == "Sustained/global deviation"


def test_end_to_end_pipeline_and_known_anomalies(tmp_path):
    data = tmp_path / "synthetic.csv"
    build_synthetic_csv(data, n=24 * 60)
    out = tmp_path / "results"
    cfg = Config(window=24, stride=6, epochs=25, hidden_dim=16, latent_dim=8, threshold_quantile=.95, seed=42)
    result, summary = run_pipeline(data, "PJME_MW", cfg, out)
    assert len(result) > 0
    assert summary["anomalous_windows"] > 0
    assert (out / "anomaly_scores.csv").exists()
    assert (out / "decision_support.csv").exists()
    assert (out / "run_summary.json").exists()
    # Every injected event should overlap at least one detected window.
    starts = result.loc[result.anomaly.eq(1), "start"]
    ends = result.loc[result.anomaly.eq(1), "end"]
    truth_ranges = [
        (pd.Timestamp("2025-01-13 12:00"), pd.Timestamp("2025-01-13 15:00")),
        (pd.Timestamp("2025-01-26 00:00"), pd.Timestamp("2025-01-26 03:00")),
        (pd.Timestamp("2025-02-07 12:00"), pd.Timestamp("2025-02-08 03:00")),
    ]
    for a, b in truth_ranges:
        overlap = ((starts <= b) & (ends >= a)).any()
        assert bool(overlap)
