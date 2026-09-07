from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler


@dataclass
class Config:
    window: int = 24
    stride: int = 6
    train_ratio: float = 0.70
    hidden_dim: int = 32
    latent_dim: int = 16
    epochs: int = 80
    batch_size: int = 64
    k1: int = 20
    k2: int = 30
    contamination: float = 0.03
    threshold_quantile: float = 0.97
    seed: int = 42


def load_energy_csv(path: str | Path, value_col: str | None = None) -> pd.DataFrame:
    """Load a PJM/Kaggle hourly-energy CSV and return hourly data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "Datetime" not in df.columns:
        raise ValueError("CSV must contain a 'Datetime' column.")
    if value_col is None:
        candidates = [c for c in df.columns if c != "Datetime"]
        if not candidates:
            raise ValueError("CSV must contain an energy-consumption column.")
        value_col = candidates[0]
    if value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found. Available: {list(df.columns)}")

    out = df[["Datetime", value_col]].copy()
    out.columns = ["Datetime", "Energy_MW"]
    out["Datetime"] = pd.to_datetime(out["Datetime"], errors="coerce")
    out["Energy_MW"] = pd.to_numeric(out["Energy_MW"], errors="coerce")
    out = out.dropna(subset=["Datetime"]).sort_values("Datetime")
    # DST can create duplicate timestamps. Mean aggregation keeps one point/hour.
    out = out.groupby("Datetime", as_index=False)["Energy_MW"].mean()
    out = out.set_index("Datetime").asfreq("h")
    out["Energy_MW"] = out["Energy_MW"].interpolate("time").ffill().bfill()
    return out.reset_index()


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create contextual time features used for decision support."""
    x = df.copy()
    dt = x["Datetime"]
    x["hour"] = dt.dt.hour
    x["dow"] = dt.dt.dayofweek
    x["hour_of_week"] = x["dow"] * 24 + x["hour"]
    x["month"] = dt.dt.month
    x["sin_hour"] = np.sin(2 * np.pi * x["hour"] / 24)
    x["cos_hour"] = np.cos(2 * np.pi * x["hour"] / 24)
    x["sin_week"] = np.sin(2 * np.pi * x["hour_of_week"] / 168)
    x["cos_week"] = np.cos(2 * np.pi * x["hour_of_week"] / 168)
    x["rolling_mean_24"] = x["Energy_MW"].rolling(24, min_periods=1).mean()
    x["rolling_std_24"] = x["Energy_MW"].rolling(24, min_periods=2).std().fillna(0)
    x["delta_1h"] = x["Energy_MW"].diff().fillna(0)
    return x


def make_windows(values: np.ndarray, window: int = 24, stride: int = 6):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if window <= 1 or stride <= 0:
        raise ValueError("window must be > 1 and stride must be > 0")
    if len(values) < window:
        raise ValueError("Not enough observations for one window")
    starts = np.arange(0, len(values) - window + 1, stride)
    windows = np.stack([values[s:s + window] for s in starts])
    return windows, starts


def robust_normal_mask(values: np.ndarray, mad_multiplier: float = 6.0) -> np.ndarray:
    """Knowledge-inspired screening: keep observations close to robust center."""
    values = np.asarray(values, dtype=float)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = max(1.4826 * mad, 1e-9)
    return np.abs(values - med) <= mad_multiplier * scale


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fit_mlp_autoencoder(train_windows: np.ndarray, cfg: Config):
    """Portable reconstruction model. It avoids mandatory PyTorch DLLs on Windows."""
    scaler = RobustScaler()
    X = scaler.fit_transform(train_windows)
    hidden = max(4, min(cfg.hidden_dim, max(8, cfg.window)))
    model = MLPRegressor(
        hidden_layer_sizes=(hidden, cfg.latent_dim, hidden),
        activation="relu",
        solver="adam",
        max_iter=cfg.epochs,
        batch_size=min(cfg.batch_size, max(1, len(X))),
        random_state=cfg.seed,
        early_stopping=False,
    )
    model.fit(X, X)
    return model, scaler


def reconstruction_scores(model, scaler, windows: np.ndarray) -> np.ndarray:
    X = scaler.transform(windows)
    recon = model.predict(X)
    return np.mean((X - recon) ** 2, axis=1)


def lof_scores(features: np.ndarray, n_neighbors: int) -> np.ndarray:
    n = len(features)
    if n < 3:
        return np.zeros(n)
    k = min(max(2, n_neighbors), n - 1)
    lof = LocalOutlierFactor(n_neighbors=k, contamination="auto")
    lof.fit_predict(features)
    return -lof.negative_outlier_factor_


def unified_score(nrrs: np.ndarray, nlds: np.ndarray, k2: int = 30):
    """Dual-density fusion inspired by the uploaded KDRF paper."""
    pair = np.column_stack([nrrs, nlds])
    fitness = lof_scores(pair, k2)
    fitness_n = minmax01(fitness)
    # Baseline fitness is 1; extra weight is applied in discordant/low-density cases.
    weight = 1.0 + fitness_n
    uas = weight * (nrrs + nlds)
    return uas, weight


def classify_event(window: np.ndarray, score: float, threshold: float) -> str:
    if score <= threshold:
        return "Normal"
    x = np.asarray(window, dtype=float)
    baseline = np.median(x)
    scale = max(np.median(np.abs(x - baseline)), 1e-9)
    z = (x - baseline) / scale
    high = np.max(z)
    low = np.min(z)
    if high >= 8 and np.mean(z > 4) < 0.35:
        return "Local spike"
    if low <= -8 and np.mean(z < -4) < 0.35:
        return "Local drop"
    if np.mean(z > 3) >= 0.35 or np.mean(z < -3) >= 0.35:
        return "Sustained/global deviation"
    return "Contextual deviation"


def decision_for_event(event_type: str) -> str:
    return {
        "Normal": "No action",
        "Local spike": "Inspect short-duration demand peak",
        "Local drop": "Check supply interruption or load loss",
        "Sustained/global deviation": "Investigate prolonged demand or operational change",
        "Contextual deviation": "Check hour/day/season context and external factors",
    }[event_type]


def run_pipeline(path: str | Path, value_col: str | None = None, cfg: Config | None = None, output_dir: str | Path = "results"):
    cfg = cfg or Config()
    np.random.seed(cfg.seed)
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = load_energy_csv(path, value_col)
    features = make_features(raw)
    values = features["Energy_MW"].to_numpy(float)
    windows, starts = make_windows(values, cfg.window, cfg.stride)
    split = max(1, int(len(windows) * cfg.train_ratio))
    train_windows = windows[:split]

    # Knowledge-inspired screening is applied to training window means.
    mask = robust_normal_mask(train_windows.mean(axis=1))
    screened = train_windows[mask]
    if len(screened) < max(10, min(100, len(train_windows))):
        screened = train_windows

    model, scaler = fit_mlp_autoencoder(screened, cfg)
    rrs = reconstruction_scores(model, scaler, windows)
    nrrs = minmax01(rrs)

    # Density is calculated from compact window statistics, reducing computational cost.
    compact = np.column_stack([
        windows.mean(axis=1),
        windows.std(axis=1),
        np.min(windows, axis=1),
        np.max(windows, axis=1),
    ])
    nlds = minmax01(lof_scores(compact, cfg.k1))
    uas, fitness = unified_score(nrrs, nlds, cfg.k2)

    train_uas = uas[:split]
    threshold = float(np.quantile(train_uas, cfg.threshold_quantile))
    labels = uas > threshold

    records = []
    for i, (start, w) in enumerate(zip(starts, windows)):
        end = start + cfg.window - 1
        score = float(uas[i])
        event = classify_event(w, score, threshold)
        records.append({
            "start": raw.iloc[start]["Datetime"],
            "end": raw.iloc[end]["Datetime"],
            "reconstruction": float(rrs[i]),
            "nRRS": float(nrrs[i]),
            "nLDS": float(nlds[i]),
            "fitness": float(fitness[i]),
            "UAS": score,
            "anomaly": int(labels[i]),
            "event_type": event,
            "decision": decision_for_event(event),
        })
    result = pd.DataFrame(records)
    result.to_csv(outdir / "anomaly_scores.csv", index=False)
    result[result["anomaly"] == 1].sort_values("UAS", ascending=False).to_csv(outdir / "decision_support.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "observations": int(len(raw)),
        "windows": int(len(windows)),
        "training_windows": int(split),
        "screened_training_windows": int(len(screened)),
        "threshold": threshold,
        "anomalous_windows": int(labels.sum()),
        "anomaly_rate": float(labels.mean()),
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return result, summary


def build_synthetic_csv(path: str | Path, n: int = 24 * 60, seed: int = 42):
    """Create a labeled synthetic dataset for repeatable tests."""
    rng = np.random.default_rng(seed)
    dt = pd.date_range("2025-01-01", periods=n, freq="h")
    t = np.arange(n)
    base = 10000 + 1200 * np.sin(2 * np.pi * t / 24) + 400 * np.sin(2 * np.pi * t / 168)
    y = base + rng.normal(0, 120, n)
    truth = np.zeros(n, dtype=int)
    # Local spike, local drop, and sustained deviation.
    y[300:304] += 5000; truth[300:304] = 1
    y[600:604] -= 3500; truth[600:604] = 1
    y[900:916] += 2800; truth[900:916] = 1
    pd.DataFrame({"Datetime": dt, "PJME_MW": y, "KnownAnomaly": truth}).to_csv(path, index=False)
    return path


def main():
    p = argparse.ArgumentParser(description="Hybrid reconstruction-density energy outlier detection")
    p.add_argument("--data", required=True)
    p.add_argument("--value-col", default=None)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--output", default="results")
    args = p.parse_args()
    cfg = Config(window=args.window, stride=args.stride, epochs=args.epochs)
    result, summary = run_pipeline(args.data, args.value_col, cfg, args.output)
    print(json.dumps(summary, indent=2, default=str))
    print("\nTop anomalies:")
    print(result.sort_values("UAS", ascending=False).head(10)[["start", "end", "UAS", "event_type", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
