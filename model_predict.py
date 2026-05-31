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
    """Загружает обученную модель и список признаков."""
    model_path = RESULTS_DIR / 'model.pkl'
    features_path = RESULTS_DIR / 'feature_names.json'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}\nСначала запустите model_train.py")
    if not features_path.exists():
        raise FileNotFoundError(f"Список признаков не найден: {features_path}\nСначала запустите model_train.py")
    
    model = joblib.load(model_path)
    with open(features_path, 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    
    print(f"Модель загружена: {type(model).__name__}")
    print(f"Признаков: {len(feature_names)}")
    return model, feature_names


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет инженерные признаки."""
    df = df.copy()
    df['ln_d']   = np.log(np.abs(df['d_m']) + 1.0)
    df['sqrt_d'] = np.sqrt(np.abs(df['d_m']) + 1.0)
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
    Прогнозирует опорное давление.
    
    Parameters
    ----------
    H : float — глубина, м (150-350)
    sigma : float — прочность кровли, МПа
    B : float — ширина ДК, м
    m_plast : float — мощность пласта, м (2-5)
    d_values : array-like — расстояния от забоя, м
    """
    if model is None or feature_names is None:
        model, feature_names = load_model()
    
    d_values = np.asarray(d_values).flatten()
    n = len(d_values)
    
    df = pd.DataFrame({
        'd_m': d_values,
        'H_m': np.full(n, H),
        'sigma_MPa': np.full(n, sigma),
        'B_m': np.full(n, B),
        'm_plast_m': np.full(n, m_plast)
    })
    df = add_engineered_features(df)
    
    for col in feature_names:
        if col.startswith('sensor_') and col not in df.columns:
            df[col] = 0
    
    X = df[feature_names].values
    return model.predict(X)


def plot_pressure_profile(d_values, predictions, H, sigma, B, m_plast, save_path=None):
    """Строит эпюру опорного давления."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(d_values, predictions, 'r-', linewidth=2.5, label='Прогноз XGBoost')
    ax.fill_between(d_values, 0, predictions, alpha=0.15, color='red')
    ax.set_xlabel('Расстояние от забоя d, м')
    ax.set_ylabel('Вертикальное давление P, МПа')
    ax.set_title(f'Прогноз опорного давления\n'
                 f'H={H} м, σ={sigma} МПа, B={B} м, m={m_plast} м')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Забой')
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
    plt.close(fig)


def main():
    print("="*60)
    print("ПРОГНОЗ ОПОРНОГО ДАВЛЕНИЯ")
    print("="*60)
    
    model, feature_names = load_model()
    
    # Пример 1: физический эксперимент
    H1, sigma1, B1, m1 = 208, 19, 4.0, 2.0
    d_vals = np.arange(-5, 41, 0.5)
    
    print(f"\nПример 1 (физ. эксперимент): H={H1}м, σ={sigma1}МПа, B={B1}м, m={m1}м")
    pred1 = predict_pressure(H1, sigma1, B1, m1, d_vals, model, feature_names)
    print(f"  P: [{pred1.min():.1f}, {pred1.max():.1f}] МПа")
    plot_pressure_profile(d_vals, pred1, H1, sigma1, B1, m1,
                         save_path=RESULTS_DIR / 'pressure_profile_phys.png')
    
    # Пример 2: новые условия
    H2, sigma2, B2, m2 = 300, 19, 4.0, 4.0
    print(f"\nПример 2 (новые условия): H={H2}м, σ={sigma2}МПа, B={B2}м, m={m2}м")
    pred2 = predict_pressure(H2, sigma2, B2, m2, d_vals, model, feature_names)
    print(f"  P: [{pred2.min():.1f}, {pred2.max():.1f}] МПа")
    plot_pressure_profile(d_vals, pred2, H2, sigma2, B2, m2,
                         save_path=RESULTS_DIR / 'pressure_profile_new.png')
    
    print("\nГотово.")


if __name__ == "__main__":
    main()
