import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
import re
import warnings
import json
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300
# Для кириллицы
plt.rcParams['font.family'] = 'DejaVu Sans'

BASE_DIR = Path(__file__).parent
LAB_DIR = BASE_DIR / 'Лаборатория'
NAGRUZKA_DIR = BASE_DIR / 'Нагрузка'
RESULTS_DIR = BASE_DIR / 'results'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# КОНСТАНТЫ (из диссертации)
GAMMA = 0.025  # МПа/м, удельный вес пород
H_PHYS = 208.0  # м, глубина для физического моделирования
SIGMA_PHYS = 19.0  # МПа, прочность непосредственной кровли
B_PHYS = 4.0  # м, ширина ДК
M_PLAST_PHYS = 2.0  # м, мощность пласта

SIGMA_NUM = 19.0  # МПа, прочность для численного моделирования (константа)
B_NUM = 4.0  # м, ширина ДК для численного моделирования (константа)


def load_physical_data() -> pd.DataFrame:
    """
    Загружает и обрабатывает данные физического моделирования из coefs.xlsx.
    
    Returns
    -------
    pd.DataFrame с колонками: d_m, H_m, sigma_MPa, B_m, m_plast_m, sensor_id, stage, K, P_MPa
    """
    file_path = LAB_DIR / 'coefs.xlsx'
    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    print(f"Загрузка физических данных: {file_path}")
    
    # Читаем ВЕСЬ файл в "сыром" виде (нет фиксированной структуры)
    raw = pd.read_excel(file_path, header=None)
    print(f"  Сырых строк: {len(raw)}")
    
    records = []
    current_stage = None
    
    for idx, row in raw.iterrows():
        # Проверяем, не заголовок ли это этапа
        col_c = row.iloc[2] if len(row) > 2 else None  # номер датчика
        col_d = row.iloc[3] if len(row) > 3 else None  # значение или заголовок

        # Проверка: это заголовок этапа?
        is_stage_header = False

        if isinstance(col_d, str) and 'Этап' in col_d:
            # Старый формат: "Этап 2, коэф.конц. напряжений"
            match = re.search(r'Этап\s+(\d+)', col_d)
            if match:
                current_stage = int(match.group(1))
                is_stage_header = True
        elif isinstance(col_d, (int, float)) and not pd.isna(col_d):
            # Новый формат: просто число (5, 6, 7...)
            # Проверяем, что колонка C (номер датчика) пустая
            if pd.isna(col_c) or col_c == '' or col_c is None:
                current_stage = int(col_d)
                is_stage_header = True

        if is_stage_header:
            continue
        
        # Пропускаем строки без данных
        if pd.isna(row.iloc[2]) and pd.isna(row.iloc[3]):
            continue
        
        try:
            sensor_id = int(row.iloc[2])
            distance_model_cm = float(row.iloc[1])  # столбец B
            K = float(row.iloc[3])  # столбец D
            
            # Пересчёт расстояния: см → м, масштаб 1:50
            d_m = distance_model_cm / 100.0 * 50.0
            
            # Пересчёт давления: P = K * γ * H
            P_MPa = K * GAMMA * H_PHYS
            
            records.append({
                'd_m': d_m,
                'H_m': H_PHYS,
                'sigma_MPa': SIGMA_PHYS,
                'B_m': B_PHYS,
                'm_plast_m': M_PLAST_PHYS,
                'sensor_id': f'D{sensor_id}',
                'stage': current_stage,
                'K': K,
                'P_MPa': P_MPa
            })
        except (ValueError, TypeError):
            continue
    
    df = pd.DataFrame(records)
    print(f"  Загружено точек: {len(df)}")
    print(f"  Этапы: {sorted(df['stage'].unique())}")
    print(f"  Датчики: {sorted(df['sensor_id'].unique())}")
    print(f"  d_m: [{df['d_m'].min():.2f}, {df['d_m'].max():.2f}] м")
    print(f"  P_MPa: [{df['P_MPa'].min():.2f}, {df['P_MPa'].max():.2f}] МПа")
    
    return df


def parse_filename(filename: str) -> dict:
    """
    Извлекает параметры из имени файла.
    Пример: "3м гл 150м.xlsx" → {'m_plast_m': 3.0, 'H_m': 150.0}
    """
    match = re.search(r'(\d+)\s*м\s*гл\s*(\d+)\s*м', filename, re.IGNORECASE)
    if match:
        return {
            'm_plast_m': float(match.group(1)),
            'H_m': float(match.group(2))
        }
    return None


def load_single_numerical_file(file_path: Path) -> pd.DataFrame:
    """
    Загружает ОДИН файл численного моделирования.
    """
    params = parse_filename(file_path.name)
    if params is None:
        print(f"  [!] Не удалось распарсить имя файла: {file_path.name}, пропущен")
        return pd.DataFrame()
    
    try:
        # Читаем, начиная со строки 8 (0-based index = 7)
        raw = pd.read_excel(file_path, header=None, skiprows=7)
    except Exception as e:
        print(f"  [!] Ошибка чтения {file_path.name}: {e}")
        return pd.DataFrame()
    
    records = []
    num_stages = 24  # Stage 1..24
    cols_per_stage = 4  # X, Y, Distance [m], Sigma One [MPa]
    
    for stage_idx in range(num_stages):
        start_col = stage_idx * cols_per_stage
        # Колонки: X, Y, Distance, Sigma One
        if start_col + 3 >= raw.shape[1]:
            break
        
        dist_col = start_col + 2  # Distance [m]
        sigma_col = start_col + 3  # Sigma One [MPa]
        
        for _, row in raw.iterrows():
            try:
                d_m = float(row.iloc[dist_col])
                p_mpa = float(row.iloc[sigma_col])
                
                if pd.isna(d_m) or pd.isna(p_mpa):
                    continue
                
                records.append({
                    'd_m': d_m,
                    'H_m': params['H_m'],
                    'sigma_MPa': SIGMA_NUM,
                    'B_m': B_NUM,
                    'm_plast_m': params['m_plast_m'],
                    'stage': stage_idx + 1,
                    'P_MPa': p_mpa
                })
            except (ValueError, TypeError, IndexError):
                continue
    
    df = pd.DataFrame(records)
    return df


def load_numerical_data(exclude_file: str = None) -> pd.DataFrame:
    """
    Загружает ВСЕ файлы численного моделирования из папки Нагрузка.
    
    Parameters
    ----------
    exclude_file : str or None
        Имя файла, который нужно ИСКЛЮЧИТЬ (для слепой валидации)
    
    Returns
    -------
    pd.DataFrame
    """
    all_data = []
    files = sorted(NAGRUZKA_DIR.glob('*.xlsx'))
    print(f"Найдено файлов численного моделирования: {len(files)}")
    
    for file_path in files:
        # Пропускаем временные файлы Excel и файлы пояснений
        if file_path.name.startswith('~$') or 'Пояснение' in file_path.name or 'пояснение' in file_path.name:
            print(f"  Пропущен: {file_path.name}")
            continue
        
        if exclude_file and file_path.name == exclude_file:
            print(f"  ИСКЛЮЧЁН (слепая выборка): {file_path.name}")
            continue
        
        df = load_single_numerical_file(file_path)
        if len(df) > 0:
            all_data.append(df)
            params = parse_filename(file_path.name)
            print(f"  {file_path.name}: {len(df)} точек, "
                  f"m_plast={params['m_plast_m']:.0f}м, H={params['H_m']:.0f}м")
    
    result = pd.concat(all_data, ignore_index=True)
    print(f"Всего загружено численных точек: {len(result)}")
    return result


def load_blind_data() -> tuple:
    """
    Выбирает ОДИН файл для слепой валидации и загружает его.
    Использует файл "3м гл 250м.xlsx" — промежуточная глубина 250 м,
    которая хорошо подходит для проверки генерализации.
    """
    blind_filename = "3м гл 250м.xlsx"
    blind_path = NAGRUZKA_DIR / blind_filename
    
    if not blind_path.exists():
        # Запасной вариант — любой файл с 250 м
        files = sorted(NAGRUZKA_DIR.glob('*.xlsx'))
        blind_path = None
        for f in files:
            if '250м' in f.name and 'Пояснение' not in f.name:
                blind_path = f
                blind_filename = f.name
                break
        
        if blind_path is None:
            print("[!] Не удалось найти файл с глубиной 250 м для слепой валидации")
            return pd.DataFrame(), None
    
    df_blind = load_single_numerical_file(blind_path)
    print(f"\nСлепая выборка: {blind_filename}, {len(df_blind)} точек")
    params = parse_filename(blind_filename)
    if params:
        print(f"  Параметры: m_plast={params['m_plast_m']:.0f}м, H={params['H_m']:.0f}м")
    
    return df_blind, blind_filename


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет инженерные признаки на основе расстояния d_m."""
    df = df.copy()
    df['ln_d']   = np.log(np.abs(df['d_m']) + 1.0)
    df['sqrt_d'] = np.sqrt(np.abs(df['d_m']) + 1.0)
    df['inv_d']  = 1.0 / (np.abs(df['d_m']) + 0.1)
    return df


def add_sensor_onehot(df: pd.DataFrame) -> tuple:
    """
    Создаёт one-hot encoding для датчиков.
    Если колонка sensor_id отсутствует — все one-hot колонки = 0.
    """
    sensor_list = ['D1','D2','D3','D4','D5','D6','D7','D8']
    sensor_cols = [f'sensor_{s}' for s in sensor_list]
    
    if 'sensor_id' not in df.columns:
        for col in sensor_cols:
            df[col] = 0
        return df, sensor_cols
    
    for i, s in enumerate(sensor_list):
        df[f'sensor_{s}'] = (df['sensor_id'] == s).astype(int)
    
    return df, sensor_cols


def main():
    print("="*60)
    print("ГИБРИДНАЯ СУРРОГАТНАЯ МОДЕЛЬ XGBOOST")
    print("Прогнозирование опорного давления")
    print("="*60)
    
    # Шаг 0: Загрузка слепой выборки и определение исключаемого файла
    print("\n[0/9] Подготовка слепой выборки...")
    df_blind, blind_filename = load_blind_data()
    
    # Шаг 1: Загрузка данных
    print("\n[1/9] Загрузка данных...")
    df_phys = load_physical_data()
    df_num  = load_numerical_data(exclude_file=blind_filename)
    
    if len(df_num) == 0:
        print("ОШИБКА: не загружено ни одной точки численного моделирования!")
        return
    
    # Шаг 2: Инженерные признаки
    print("\n[2/9] Создание инженерных признаков...")
    df_phys = add_engineered_features(df_phys)
    df_num  = add_engineered_features(df_num)
    
    # Шаг 3: One-hot encoding датчиков
    print("\n[3/9] Кодирование датчиков...")
    df_phys, sensor_cols = add_sensor_onehot(df_phys)
    for col in sensor_cols:
        if col not in df_num.columns:
            df_num[col] = 0
    
    # Шаг 4: Объединение в гибридный датасет
    print("\n[4/9] Формирование гибридного датасета...")
    feature_cols = ['d_m','H_m','sigma_MPa','B_m','m_plast_m',
                    'ln_d','sqrt_d','inv_d'] + sensor_cols
    target_col = 'P_MPa'
    
    # Проверяем, что все колонки есть
    for col in feature_cols:
        if col not in df_phys.columns:
            print(f"[!] Колонка {col} отсутствует в физических данных, заполняю нулями")
            df_phys[col] = 0
        if col not in df_num.columns:
            print(f"[!] Колонка {col} отсутствует в численных данных, заполняю нулями")
            df_num[col] = 0
    
    df_hybrid = pd.concat([
        df_phys[feature_cols + [target_col, 'stage']],
        df_num[feature_cols + [target_col, 'stage']]
    ], ignore_index=True)
    
    print(f"Гибридный датасет: {len(df_hybrid)} точек")
    print(f"  Физическое моделирование: {len(df_phys)} точек")
    print(f"  Численное моделирование:  {len(df_num)} точек")
    print(f"  Признаков: {len(feature_cols)}")
    print(f"  Диапазон H: [{df_hybrid['H_m'].min():.0f}, {df_hybrid['H_m'].max():.0f}] м")
    print(f"  Диапазон m_plast: [{df_hybrid['m_plast_m'].min():.0f}, {df_hybrid['m_plast_m'].max():.0f}] м")
    
    # Очистка гибридного датасета от NaN и inf
    df_hybrid = df_hybrid.dropna(subset=['P_MPa', 'd_m'])
    df_hybrid = df_hybrid[~df_hybrid.isin([np.inf, -np.inf]).any(axis=1)]
    print(f"  После очистки: {len(df_hybrid)} точек")
    
    # Шаг 5: Разделение train/test
    print("\n[5/9] Разделение на train/test...")
    df_hybrid['H_bin'] = pd.cut(df_hybrid['H_m'], bins=5, labels=False)
    
    X = df_hybrid[feature_cols].values
    y = df_hybrid[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42,
        stratify=df_hybrid['H_bin']
    )
    print(f"  Train: {len(X_train)} точек")
    print(f"  Test:  {len(X_test)} точек")
    
    # Шаг 6: Обучение моделей
    print("\n[6/9] Обучение моделей...")
    
    # Baseline: LinearRegression
    print("  Обучаю LinearRegression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Baseline: Polynomial (degree=3)
    print("  Обучаю PolynomialRegression (degree=3)...")
    poly_pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    poly_pipe.fit(X_train, y_train)
    y_pred_poly = poly_pipe.predict(X_test)
    
    # XGBoost
    print("  Обучаю XGBoost с GridSearchCV...")
    xgb = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0]
    }
    
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    print(f"  Лучшие параметры: {grid.best_params_}")
    print(f"  Лучший R2 (CV):   {grid.best_score_:.4f}")
    
    y_pred_xgb = best_model.predict(X_test)
    
    # Шаг 7: Оценка
    print("\n[7/9] Оценка моделей...")
    
    def evaluate(y_true, y_pred, name):
        return {
            'Модель': name,
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    results = [
        evaluate(y_test, y_pred_lr, 'Линейная регрессия'),
        evaluate(y_test, y_pred_poly, 'Полиномиальная (deg=3)'),
        evaluate(y_test, y_pred_xgb, 'XGBoost')
    ]
    
    df_results = pd.DataFrame(results)
    df_results = df_results.round({'R2': 4, 'MAE': 2, 'RMSE': 2})
    print("\n" + df_results.to_string(index=False))
    df_results.to_csv(RESULTS_DIR / 'results_summary.csv', index=False)
    
    # Шаг 8: Графики
    print("\n[8/9] Построение графиков...")
    
    # --- График 1: Predicted vs Actual (XGBoost) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Только точки XGBoost
    ax.scatter(y_test, y_pred_xgb, alpha=0.3, edgecolors='black', 
               linewidth=0.2, s=30, c='steelblue', label='XGBoost')
    
    # Фиксированные пределы
    ax.set_xlim(-25, 75)
    ax.set_ylim(-25, 75)
    
    # Линия y=x
    ax.plot([-25, 75], [-25, 75], 'r--', linewidth=2, label='y = x')
    
    ax.set_xlabel('Actual pressure, MPa')
    ax.set_ylabel('Predicted pressure, MPa')
    ax.set_title(f'XGBoost: predicted vs actual\n'
                 f'R² = {r2_score(y_test, y_pred_xgb):.4f}, '
                 f'MAE = {mean_absolute_error(y_test, y_pred_xgb):.2f} MPa')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'fig1_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / 'fig1_predicted_vs_actual.svg', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] fig1_predicted_vs_actual.png/.svg")
    
    # --- График 2: Residuals ---
    residuals = y_pred_xgb - y_test
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, residuals, alpha=0.3, edgecolors='black', linewidth=0.2, s=30)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Actual pressure, MPa')
    ax.set_ylabel('Residuals, MPa')
    ax.set_title('XGBoost: residual plot')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'fig2_residuals.png', dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / 'fig2_residuals.svg', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] fig2_residuals.png/.svg")
    
    # --- График 3: Слепая валидация ---
    if len(df_blind) > 0:
        print("  Выполняю слепую валидацию...")
        df_blind = add_engineered_features(df_blind)
        for col in sensor_cols:
            df_blind[col] = 0
        
        X_blind = df_blind[feature_cols].values
        y_blind_true = df_blind['P_MPa'].values
        y_blind_pred = best_model.predict(X_blind)
        
        # Сортируем по расстоянию для красивого графика
        sort_idx = np.argsort(df_blind['d_m'].values)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df_blind['d_m'].values[sort_idx], y_blind_pred[sort_idx],
                'r-', linewidth=2, alpha=0.8, label='XGBoost prediction')
        ax.scatter(df_blind['d_m'].values[sort_idx], y_blind_true[sort_idx],
                   s=15, c='blue', marker='o', alpha=0.5, label='FLAC3D (reference)')
        ax.set_xlabel('Distance from face d, m')
        ax.set_ylabel('Vertical pressure P, MPa')
        params = parse_filename(blind_filename)
        ax.set_title(
            f'Blind validation: seam thickness m = {params["m_plast_m"]:.0f} m, '
            f'depth H = {params["H_m"]:.0f} m\n(data excluded from training)'
        )
        ax.set_xlim(-5, 45)
        ax.set_ylim(-5, 60)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / 'fig3_blind_validation.png', dpi=300, bbox_inches='tight')
        fig.savefig(RESULTS_DIR / 'fig3_blind_validation.svg', bbox_inches='tight')
        fig.savefig(RESULTS_DIR / 'figure4_blind_validation.png', dpi=300, bbox_inches='tight')
        fig.savefig(RESULTS_DIR / 'figure4_blind_validation.svg', bbox_inches='tight')
        plt.close(fig)
        print("  [OK] fig3_blind_validation.png/.svg")
        print("  [OK] figure4_blind_validation.png/.svg")
        
        blind_r2 = r2_score(y_blind_true, y_blind_pred)
        blind_mae = mean_absolute_error(y_blind_true, y_blind_pred)
        print(f"  Слепая выборка: R2={blind_r2:.4f}, MAE={blind_mae:.2f} МПа")
    else:
        print("[!] Слепая выборка пуста, график не построен")
    
    # --- График 4: Feature Importance ---
    importance = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feat_imp['feature'], feat_imp['importance'], color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Feature importance')
    ax.set_title('XGBoost: feature importance')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'fig4_feature_importance.png', dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / 'fig4_feature_importance.svg', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] fig4_feature_importance.png/.svg")
    
    # --- График 5: Пространственно-временная тепловая карта ---
    print("  Строю пространственно-временную тепловую карту...")

    # Параметры физического эксперимента
    # Шаг подвигания забоя: 1 м (натурный) — из диссертации
    STEP = 1.0  # м

    # Координаты точек массива (фиксированные в пространстве)
    x_points = np.linspace(-5, 40, 90)  # 90 точек вдоль выемочного столба

    # Этапы подвигания
    stages = sorted(df_phys['stage'].unique())

    # Координата забоя на каждом этапе:
    # забой стартует с X=0 и движется вперёд на STEP каждый этап
    # X_face(stage) = X_face_start + (stage - stage_min) * STEP
    stage_min = min(stages)
    x_face_start = 0.0  # начальное положение забоя (м), уточни по данным

    heatmap_data = np.zeros((len(stages), len(x_points)))

    for i, stage in enumerate(stages):
        # Текущее положение забоя
        x_face = x_face_start + (stage - stage_min) * STEP
        
        for j, x_point in enumerate(x_points):
            # Расстояние от точки массива до забоя
            d_val = x_point - x_face
            
            row_features = {
                'd_m': d_val,
                'H_m': H_PHYS,
                'sigma_MPa': SIGMA_PHYS,
                'B_m': B_PHYS,
                'm_plast_m': M_PLAST_PHYS,
                'ln_d': np.log(np.abs(d_val) + 1.0),
                'sqrt_d': np.sqrt(np.abs(d_val) + 1.0),
                'inv_d': 1.0 / (np.abs(d_val) + 0.1)
            }
            X_pred = np.zeros((1, len(feature_cols)))
            for k, col in enumerate(feature_cols):
                if col in row_features:
                    X_pred[0, k] = row_features[col]
                elif col.startswith('sensor_'):
                    X_pred[0, k] = 0
            
            heatmap_data[i, j] = best_model.predict(X_pred)[0]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.pcolormesh(x_points, stages, heatmap_data, cmap='RdYlBu_r', shading='auto')
    cbar = fig.colorbar(im, ax=ax, label='Vertical pressure P, MPa')
    ax.set_xlabel('Coordinate along mining panel, m')
    ax.set_ylabel('Face advance stage')
    ax.set_title(
        'Spatio-temporal evolution of abutment pressure\n'
        f'H = {H_PHYS:.0f} m, σ = {SIGMA_PHYS:.0f} MPa, '
        f'B = {B_PHYS:.1f} m, m = {M_PLAST_PHYS:.1f} m '
        '(surrogate-model reconstruction)'
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'fig5_spacetime_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / 'fig5_spacetime_heatmap.svg', bbox_inches='tight')
    fig.savefig(RESULTS_DIR / 'figure5_spacetime_heatmap.png', dpi=300, bbox_inches='tight')
    fig.savefig(RESULTS_DIR / 'figure5_spacetime_heatmap.svg', bbox_inches='tight')
    plt.close(fig)
    print("  [OK] fig5_spacetime_heatmap.png/.svg")
    print("  [OK] figure5_spacetime_heatmap.png/.svg")
    
    # Шаг 9: Сохранение модели
    print("\n[9/9] Сохранение модели...")
    joblib.dump(best_model, RESULTS_DIR / 'model.pkl')
    with open(RESULTS_DIR / 'feature_names.json', 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"  Модель: {RESULTS_DIR / 'model.pkl'}")
    print(f"  Признаки: {RESULTS_DIR / 'feature_names.json'}")
    
    # Итог
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    best_row = df_results[df_results['Модель'] == 'XGBoost'].iloc[0]
    print(f"XGBoost: R2={best_row['R2']:.4f}, MAE={best_row['MAE']:.2f} МПа, RMSE={best_row['RMSE']:.2f} МПа")
    print(f"Гибридный датасет: {len(df_hybrid)} точек")
    print(f"Графики: {RESULTS_DIR}")
    print(f"Модель: {RESULTS_DIR}")
    
    # Сохранение итогового резюме в файл
    with open(RESULTS_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write("ГИБРИДНАЯ СУРРОГАТНАЯ МОДЕЛЬ XGBOOST\n")
        f.write("Прогнозирование опорного давления\n")
        f.write("="*60 + "\n\n")
        f.write(f"Гибридный датасет: {len(df_hybrid)} точек\n")
        f.write(f"  Физическое моделирование: {len(df_phys)} точек\n")
        f.write(f"  Численное моделирование:  {len(df_num)} точек\n\n")
        f.write(f"Лучшие гиперпараметры: {grid.best_params_}\n\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\nСлепая выборка: ")
        if len(df_blind) > 0:
            f.write(f"R2={blind_r2:.4f}, MAE={blind_mae:.2f} МПа\n")
        else:
            f.write("не проводилась\n")


if __name__ == "__main__":
    main()
