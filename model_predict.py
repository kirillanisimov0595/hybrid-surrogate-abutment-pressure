import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import json
from pathlib import Path
from typing import Union, List

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12


def load_model():
    """Loads the trained model and feature list."""
    model_path = RESULTS_DIR / 'model.pkl'
    features_path = RESULTS_DIR / 'feature_names.json'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}\nRun model_train.py first")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature list not found: {features_path}\nRun model_train.py first")
    
    model = joblib.load(model_path)
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Number of features: {len(feature_names)}")
    return model, feature_names


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered features for geomechanical modeling."""
    df = df.copy()
    # Natural log transform for distance from longwall face
    df['ln_d']   = np.log(np.abs(df['d_m']) + 1.0)
    # Square root transform to capture nonlinear stress decay
    df['sqrt_d'] = np.sqrt(np.abs(df['d_m']) + 1.0)
    # Inverse distance weighting for near-face stress concentration
    df['inv_d']  = 1.0 / (np.abs(df['d_m']) + 0.1)
    return df


def predict_pressure(
    H: float,
    sigma: float,
    B: float,
    m_plast: float,
    d_values: Union[List[float], np.ndarray],
    model=None,
    feature_names: List[str] = None
) -> np.ndarray:
    """
    Predicts abutment pressure distribution ahead of the longwall face.
    
    Parameters
    ----------
    H : float — overburden depth, m (150-350)
    sigma : float — roof rock strength, MPa
    B : float — longwall panel width, m
    m_plast : float — coal seam thickness, m (2-5)
    d_values : array-like — distances from the longwall face, m
    
    Returns
    -------
    np.ndarray — predicted vertical stress values, MPa
    """
    if model is None or feature_names is None:
        model, feature_names = load_model()
    
    d_values = np.asarray(d_values).flatten()
    n = len(d_values)
    
    # Construct input dataframe with mining and geological parameters
    df = pd.DataFrame({
        'd_m': d_values,
        'H_m': np.full(n, H),
        'sigma_MPa': np.full(n, sigma),
        'B_m': np.full(n, B),
        'm_plast_m': np.full(n, m_plast)
    })
    df = add_engineered_features(df)
    
    # Handle optional sensor columns that may not be present in base model
    for col in feature_names:
        if col.startswith('sensor_') and col not in df.columns:
            df[col] = 0
    
    X = df[feature_names].values
    return model.predict(X)


def plot_pressure_profile(d_values, predictions, H, sigma, B, m_plast, save_path=None):
    """Plots the abutment pressure distribution profile."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(d_values, predictions, 'r-', linewidth=2.5, label='XGBoost prediction')
    ax.fill_between(d_values, 0, predictions, alpha=0.15, color='red')
    ax.set_xlabel('Distance from longwall face d, m')
    ax.set_ylabel('Vertical stress P, MPa')
    ax.set_title(f'Abutment pressure prediction\n'
                 f'H={H} m, σ={sigma} MPa, B={B} m, m={m_plast} m')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Longwall face')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.close(fig)


def main():
    print("="*60)
    print("ABUTMENT PRESSURE PREDICTION")
    print("="*60)
    
    model, feature_names = load_model()
    
    # Case 1: Physical modeling experiment
    H1, sigma1, B1, m1 = 208, 19, 4.0, 2.0
    d_vals = np.arange(-5, 41, 0.5)
    
    print(f"\nCase 1 (physical experiment): H={H1}m, σ={sigma1}MPa, B={B1}m, m={m1}m")
    pred1 = predict_pressure(H1, sigma1, B1, m1, d_vals, model, feature_names)
    print(f"  P: [{pred1.min():.1f}, {pred1.max():.1f}] MPa")
    plot_pressure_profile(d_vals, pred1, H1, sigma1, B1, m1,
                         save_path=RESULTS_DIR / 'pressure_profile_phys.png')
    
    # Case 2: Alternative mining conditions
    H2, sigma2, B2, m2 = 300, 19, 4.0, 4.0
    print(f"\nCase 2 (alternative conditions): H={H2}m, σ={sigma2}MPa, B={B2}m, m={m2}m")
    pred2 = predict_pressure(H2, sigma2, B2, m2, d_vals, model, feature_names)
    print(f"  P: [{pred2.min():.1f}, {pred2.max():.1f}] MPa")
    plot_pressure_profile(d_vals, pred2, H2, sigma2, B2, m2,
                         save_path=RESULTS_DIR / 'pressure_profile_new.png')
    
    print("\nDone.")


if __name__ == "__main__":
    main()
