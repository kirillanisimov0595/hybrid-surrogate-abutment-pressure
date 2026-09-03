"""Regenerate manuscript figures with English labels."""

from pathlib import Path

import joblib
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from model_train import (
    B_PHYS,
    H_PHYS,
    M_PLAST_PHYS,
    RESULTS_DIR,
    SIGMA_PHYS,
    add_engineered_features,
    add_sensor_onehot,
    load_blind_data,
    load_numerical_data,
    load_physical_data,
    parse_filename,
)

matplotlib.use("Agg")
plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "DejaVu Sans"


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")


def load_trained_artifacts():
    model_path = RESULTS_DIR / "model.pkl"
    features_path = RESULTS_DIR / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run model_train.py first."
        )

    model = joblib.load(model_path)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return model, feature_cols


def prepare_datasets():
    df_blind, blind_filename = load_blind_data()
    df_phys = load_physical_data()
    df_num = load_numerical_data(exclude_file=blind_filename)

    df_phys = add_engineered_features(df_phys)
    df_num = add_engineered_features(df_num)
    df_phys, sensor_cols = add_sensor_onehot(df_phys)

    for col in sensor_cols:
        if col not in df_num.columns:
            df_num[col] = 0

    feature_cols = [
        "d_m",
        "H_m",
        "sigma_MPa",
        "B_m",
        "m_plast_m",
        "ln_d",
        "sqrt_d",
        "inv_d",
    ] + sensor_cols

    for col in feature_cols:
        if col not in df_phys.columns:
            df_phys[col] = 0
        if col not in df_num.columns:
            df_num[col] = 0

    df_hybrid = pd.concat(
        [
            df_phys[feature_cols + ["P_MPa", "stage"]],
            df_num[feature_cols + ["P_MPa", "stage"]],
        ],
        ignore_index=True,
    )
    df_hybrid = df_hybrid.dropna(subset=["P_MPa", "d_m"])
    df_hybrid = df_hybrid[~df_hybrid.isin([np.inf, -np.inf]).any(axis=1)]
    df_hybrid["H_bin"] = pd.cut(df_hybrid["H_m"], bins=5, labels=False)

    X = df_hybrid[feature_cols].values
    y = df_hybrid["P_MPa"].values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=df_hybrid["H_bin"]
    )

    return df_phys, df_blind, blind_filename, feature_cols, sensor_cols, X_test, y_test


def plot_predicted_vs_actual(model, X_test, y_test, output_dir: Path) -> None:
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        y_test,
        y_pred,
        alpha=0.3,
        edgecolors="black",
        linewidth=0.2,
        s=30,
        c="steelblue",
        label="XGBoost",
    )
    ax.set_xlim(-25, 75)
    ax.set_ylim(-25, 75)
    ax.plot([-25, 75], [-25, 75], "r--", linewidth=2, label="y = x")
    ax.set_xlabel("Actual pressure, MPa")
    ax.set_ylabel("Predicted pressure, MPa")
    ax.set_title(
        "XGBoost: predicted vs actual\n"
        f"R² = {r2_score(y_test, y_pred):.4f}, "
        f"MAE = {mean_absolute_error(y_test, y_pred):.2f} MPa"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir / "fig1_predicted_vs_actual")
    plt.close(fig)


def plot_residuals(model, X_test, y_test, output_dir: Path) -> None:
    y_pred = model.predict(X_test)
    residuals = y_pred - y_test

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, residuals, alpha=0.3, edgecolors="black", linewidth=0.2, s=30)
    ax.axhline(y=0, color="r", linestyle="--", linewidth=2)
    ax.set_xlabel("Actual pressure, MPa")
    ax.set_ylabel("Residuals, MPa")
    ax.set_title("XGBoost: residual plot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir / "fig2_residuals")
    plt.close(fig)


def plot_blind_validation(model, df_blind, blind_filename, feature_cols, sensor_cols, output_dir: Path) -> None:
    if len(df_blind) == 0:
        print("[!] Blind dataset is empty; figure 4 was not generated.")
        return

    df_blind = add_engineered_features(df_blind.copy())
    for col in sensor_cols:
        df_blind[col] = 0

    X_blind = df_blind[feature_cols].values
    y_true = df_blind["P_MPa"].values
    y_pred = model.predict(X_blind)
    sort_idx = np.argsort(df_blind["d_m"].values)
    params = parse_filename(blind_filename) or {"m_plast_m": 3.0, "H_m": 250.0}

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        df_blind["d_m"].values[sort_idx],
        y_pred[sort_idx],
        "r-",
        linewidth=2,
        alpha=0.8,
        label="XGBoost prediction",
    )
    ax.scatter(
        df_blind["d_m"].values[sort_idx],
        y_true[sort_idx],
        s=15,
        c="blue",
        marker="o",
        alpha=0.5,
        label="FLAC3D (reference)",
    )
    ax.set_xlabel("Distance from face d, m")
    ax.set_ylabel("Vertical pressure P, MPa")
    ax.set_title(
        f"Blind validation: seam thickness m = {params['m_plast_m']:.0f} m, "
        f"depth H = {params['H_m']:.0f} m\n(data excluded from training)"
    )
    ax.set_xlim(-5, 45)
    ax.set_ylim(-5, 60)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output_dir / "fig3_blind_validation")
    save_figure(fig, output_dir / "figure4_blind_validation")
    plt.close(fig)


def plot_feature_importance(model, feature_cols, output_dir: Path) -> None:
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": feature_cols, "importance": importance}).sort_values(
        "importance", ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        feat_imp["feature"],
        feat_imp["importance"],
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Feature importance")
    ax.set_title("XGBoost: feature importance")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    save_figure(fig, output_dir / "fig4_feature_importance")
    plt.close(fig)


def plot_spacetime_heatmap(model, df_phys, feature_cols, output_dir: Path) -> None:
    step_m = 1.0
    x_points = np.linspace(-5, 40, 90)
    stages = sorted(df_phys["stage"].unique())
    stage_min = min(stages)
    x_face_start = 0.0
    heatmap_data = np.zeros((len(stages), len(x_points)))

    for i, stage in enumerate(stages):
        x_face = x_face_start + (stage - stage_min) * step_m
        for j, x_point in enumerate(x_points):
            d_val = x_point - x_face
            row_features = {
                "d_m": d_val,
                "H_m": H_PHYS,
                "sigma_MPa": SIGMA_PHYS,
                "B_m": B_PHYS,
                "m_plast_m": M_PLAST_PHYS,
                "ln_d": np.log(np.abs(d_val) + 1.0),
                "sqrt_d": np.sqrt(np.abs(d_val) + 1.0),
                "inv_d": 1.0 / (np.abs(d_val) + 0.1),
            }
            x_pred = np.zeros((1, len(feature_cols)))
            for k, col in enumerate(feature_cols):
                if col in row_features:
                    x_pred[0, k] = row_features[col]
                elif col.startswith("sensor_"):
                    x_pred[0, k] = 0
            heatmap_data[i, j] = model.predict(x_pred)[0]

    fig, ax = plt.subplots(figsize=(14, 8))
    mesh = ax.pcolormesh(x_points, stages, heatmap_data, cmap="RdYlBu_r", shading="auto")
    fig.colorbar(mesh, ax=ax, label="Vertical pressure P, MPa")
    ax.set_xlabel("Coordinate along mining panel, m")
    ax.set_ylabel("Face advance stage")
    ax.set_title(
        "Spatio-temporal evolution of abutment pressure\n"
        f"H = {H_PHYS:.0f} m, σ = {SIGMA_PHYS:.0f} MPa, "
        f"B = {B_PHYS:.1f} m, m = {M_PLAST_PHYS:.1f} m "
        "(surrogate-model reconstruction)"
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "fig5_spacetime_heatmap")
    save_figure(fig, output_dir / "figure5_spacetime_heatmap")
    plt.close(fig)


def main() -> None:
    output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading trained model...")
    model, feature_cols = load_trained_artifacts()

    print("Loading datasets...")
    df_phys, df_blind, blind_filename, feature_cols, sensor_cols, X_test, y_test = prepare_datasets()

    print("Generating English figures...")
    plot_predicted_vs_actual(model, X_test, y_test, output_dir)
    plot_residuals(model, X_test, y_test, output_dir)
    plot_blind_validation(model, df_blind, blind_filename, feature_cols, sensor_cols, output_dir)
    plot_feature_importance(model, feature_cols, output_dir)
    plot_spacetime_heatmap(model, df_phys, feature_cols, output_dir)

    print("\nDone. Files saved to:", output_dir)
    print("- fig1_predicted_vs_actual.png/.svg")
    print("- fig2_residuals.png/.svg")
    print("- fig3_blind_validation.png/.svg")
    print("- figure4_blind_validation.png/.svg  (manuscript Figure 4)")
    print("- fig4_feature_importance.png/.svg")
    print("- fig5_spacetime_heatmap.png/.svg")
    print("- figure5_spacetime_heatmap.png/.svg  (manuscript Figure 5)")


if __name__ == "__main__":
    main()
