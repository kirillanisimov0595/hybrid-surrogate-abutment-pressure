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
# Font settings for international compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'

BASE_DIR = Path(__file__).parent
LAB_DIR = BASE_DIR / 'Labs'
PRESSURE_DIR = BASE_DIR / 'Pressure'
RESULTS_DIR = BASE_DIR / 'results'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CONSTANTS
GAMMA = 0.025  # MPa/m, unit weight of overburden rocks
H_PHYS = 208.0  # m, depth for physical modeling
SIGMA_PHYS = 19.0  # MPa, immediate roof strength
B_PHYS = 4.0  # m, longwall panel width
M_PLAST_PHYS = 2.0  # m, coal seam thickness

SIGMA_NUM = 19.0  # MPa, strength for numerical modeling (constant)
B_NUM = 4.0  # m, longwall panel width for numerical modeling (constant)


def load_physical_data() -> pd.DataFrame:
    """
    Loads and processes physical modeling data from coefs.xlsx.
    
    Returns
    -------
    pd.DataFrame with columns: d_m, H_m, sigma_MPa, B_m, m_plast_m, sensor_id, stage, K, P_MPa
    """
    file_path = LAB_DIR / 'coefs.xlsx'
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading physical modeling data: {file_path}")
    
    # Read the ENTIRE file in raw format (no fixed structure)
    raw = pd.read_excel(file_path, header=None)
    print(f"  Raw rows: {len(raw)}")
    
    records = []
    current_stage = None
    
    for idx, row in raw.iterrows():
        # Check if this is a stage header row
        col_c = row.iloc[2] if len(row) > 2 else None  # sensor number
        col_d = row.iloc[3] if len(row) > 3 else None  # value or header

        # Check: is this a stage header?
        is_stage_header = False

        if isinstance(col_d, str) and 'Этап' in col_d:
            # Legacy format: "Этап 2, коэф.конц. напряжений" (Stage 2, stress concentration factor)
            match = re.search(r'Этап\s+(\d+)', col_d)
            if match:
                current_stage = int(match.group(1))
                is_stage_header = True
        elif isinstance(col_d, (int, float)) and not pd.isna(col_d):
            # New format: just a number (5, 6, 7...)
            # Check that column C (sensor number) is empty
            if pd.isna(col_c) or col_c == '' or col_c is None:
                current_stage = int(col_d)
                is_stage_header = True

        if is_stage_header:
            continue
        
        # Skip rows without data
        if pd.isna(row.iloc[2]) and pd.isna(row.iloc[3]):
            continue
        
        try:
            sensor_id = int(row.iloc[2])
            distance_model_cm = float(row.iloc[1])  # column B
            K = float(row.iloc[3])  # column D
            
            # Convert distance: cm → m, scale factor 1:50
            d_m = distance_model_cm / 100.0 * 50.0
            
            # Convert pressure: P = K * γ * H
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
    print(f"  Loaded data points: {len(df)}")
    print(f"  Stages: {sorted(df['stage'].unique())}")
    print(f"  Sensors: {sorted(df['sensor_id'].unique())}")
    print(f"  d_m range: [{df['d_m'].min():.2f}, {df['d_m'].max():.2f}] m")
    print(f"  P_MPa range: [{df['P_MPa'].min():.2f}, {df['P_MPa'].max():.2f}] MPa")
    
    return df


def parse_filename(filename: str) -> dict:
    """
    Extracts parameters from filename.
    Example: "3м гл 150м.xlsx" → {'m_plast_m': 3.0, 'H_m': 150.0}
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
    Loads a SINGLE numerical modeling file.
    """
    params = parse_filename(file_path.name)
    if params is None:
        print(f"  [!] Failed to parse filename: {file_path.name}, skipped")
        return pd.DataFrame()
    
    try:
        # Read starting from row 8 (0-based index = 7)
        raw = pd.read_excel(file_path, header=None, skiprows=7)
    except Exception as e:
        print(f"  [!] Error reading {file_path.name}: {e}")
        return pd.DataFrame()
    
    records = []
    num_stages = 24  # Stage 1..24
    cols_per_stage = 4  # X, Y, Distance [m], Sigma One [MPa]
    
    for stage_idx in range(num_stages):
        start_col = stage_idx * cols_per_stage
        # Columns: X, Y, Distance, Sigma One
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
    Loads ALL numerical modeling files from the Pressure folder.
    
    Parameters
    ----------
    exclude_file : str or None
        Filename to EXCLUDE (for blind validation)
    
    Returns
    -------
    pd.DataFrame
    """
    all_data = []
    files = sorted(PRESSURE_DIR.glob('*.xlsx'))
    print(f"Numerical modeling files found: {len(files)}")
    
    for file_path in files:
        # Skip temporary Excel files and documentation files
        if file_path.name.startswith('~$') or 'Пояснение' in file_path.name or 'пояснение' in file_path.name:
            print(f"  Skipped: {file_path.name}")
            continue
        
        if exclude_file and file_path.name == exclude_file:
            print(f"  EXCLUDED (blind validation set): {file_path.name}")
            continue
        
        df = load_single_numerical_file(file_path)
        if len(df) > 0:
            all_data.append(df)
            params = parse_filename(file_path.name)
            print(f"  {file_path.name}: {len(df)} points, "
                  f"m_plast={params['m_plast_m']:.0f}m, H={params['H_m']:.0f}m")
    
    result = pd.concat(all_data, ignore_index=True)
    print(f"Total numerical data points loaded: {len(result)}")
    return result


def load_blind_data() -> tuple:
    """
    Selects ONE file for blind validation and loads it.
    Uses file "3м гл 250м.xlsx" — intermediate depth of 250 m,
    which is well-suited for testing generalization capability.
    """
    blind_filename = "3м гл 250м.xlsx"
    blind_path = PRESSURE_DIR / blind_filename
    
    if not blind_path.exists():
        # Fallback option — any file with 250 m depth
        files = sorted(PRESSURE_DIR.glob('*.xlsx'))
        blind_path = None
        for f in files:
            if '250м' in f.name and 'Пояснение' not in f.name:
                blind_path = f
                blind_filename = f.name
                break
        
        if blind_path is None:
            print("[!] Could not find file with 250 m depth for blind validation")
            return pd.DataFrame(), None
    
    df_blind = load_single_numerical_file(blind_path)
    print(f"\nBlind validation set: {blind_filename}, {len(df_blind)} points")
    params = parse_filename(blind_filename)
    if params:
        print(f"  Parameters: m_plast={params['m_plast_m']:.0f}m, H={params['H_m']:.0f}m")
    
    return df_blind, blind_filename


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engineered features based on distance from longwall face d_m."""
    df = df.copy()
    # Natural log transform for nonlinear stress decay with distance
    df['ln_d']   = np.log(np.abs(df['d_m']) + 1.0)
    # Square root transform to capture stress gradient near the face
    df['sqrt_d'] = np.sqrt(np.abs(df['d_m']) + 1.0)
    # Inverse distance weighting for near-field stress concentration
    df['inv_d']  = 1.0 / (np.abs(df['d_m']) + 0.1)
    return df


def add_sensor_onehot(df: pd.DataFrame) -> tuple:
    """
    Creates one-hot encoding for pressure sensors.
    If sensor_id column is absent — all one-hot columns = 0.
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
    print("HYBRID SURROGATE MODEL XGBOOST")
    print("Abutment Pressure Prediction")
    print("="*60)
    
    # Step 0: Load blind validation set and determine excluded file
    print("\n[0/9] Preparing blind validation set...")
    df_blind, blind_filename = load_blind_data()
    
    # Step 1: Data loading
    print("\n[1/9] Loading data...")
    df_phys = load_physical_data()
    df_num  = load_numerical_data(exclude_file=blind_filename)
    
    if len(df_num) == 0:
        print("ERROR: No numerical modeling data points loaded!")
        return
    
    # Step 2: Feature engineering
    print("\n[2/9] Creating engineered features...")
    df_phys = add_engineered_features(df_phys)
    df_num  = add_engineered_features(df_num)
    
    # Step 3: One-hot encoding for sensors
    print("\n[3/9] Encoding sensors...")
    df_phys, sensor_cols = add_sensor_onehot(df_phys)
    for col in sensor_cols:
        if col not in df_num.columns:
            df_num[col] = 0
    
    # Step 4: Building hybrid dataset
    print("\n[4/9] Building hybrid dataset...")
    feature_cols = ['d_m','H_m','sigma_MPa','B_m','m_plast_m',
                    'ln_d','sqrt_d','inv_d'] + sensor_cols
    target_col = 'P_MPa'
    
    # Verify all columns exist
    for col in feature_cols:
        if col not in df_phys.columns:
            print(f"[!] Column {col} missing in physical data, filling with zeros")
            df_phys[col] = 0
        if col not in df_num.columns:
            print(f"[!] Column {col} missing in numerical data, filling with zeros")
            df_num[col] = 0
    
    df_hybrid = pd.concat([
        df_phys[feature_cols + [target_col, 'stage']],
        df_num[feature_cols + [target_col, 'stage']]
    ], ignore_index=True)
    
    print(f"Hybrid dataset: {len(df_hybrid)} points")
    print(f"  Physical modeling: {len(df_phys)} points")
    print(f"  Numerical modeling: {len(df_num)} points")
    print(f"  Features: {len(feature_cols)}")
    print(f"  H range: [{df_hybrid['H_m'].min():.0f}, {df_hybrid['H_m'].max():.0f}] m")
    print(f"  m_plast range: [{df_hybrid['m_plast_m'].min():.0f}, {df_hybrid['m_plast_m'].max():.0f}] m")
    
    # Clean hybrid dataset from NaN and inf values
    df_hybrid = df_hybrid.dropna(subset=['P_MPa', 'd_m'])
    df_hybrid = df_hybrid[~df_hybrid.isin([np.inf, -np.inf]).any(axis=1)]
    print(f"  After cleaning: {len(df_hybrid)} points")
    
    # Step 5: Train/test split
    print("\n[5/9] Splitting into train/test sets...")
    df_hybrid['H_bin'] = pd.cut(df_hybrid['H_m'], bins=5, labels=False)
    
    X = df_hybrid[feature_cols].values
    y = df_hybrid[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42,
        stratify=df_hybrid['H_bin']
    )
    print(f"  Train: {len(X_train)} points")
    print(f"  Test:  {len(X_test)} points")
    
    # Step 6: Model training
    print("\n[6/9] Training models...")
    
    # Baseline: LinearRegression
    print("  Training LinearRegression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    # Baseline: Polynomial (degree=3)
    print("  Training PolynomialRegression (degree=3)...")
    poly_pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    poly_pipe.fit(X_train, y_train)
    y_pred_poly = poly_pipe.predict(X_test)
    
    # XGBoost with hyperparameter tuning
    print("  Training XGBoost with GridSearchCV...")
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
    print(f"  Best parameters: {grid.best_params_}")
    print(f"  Best R2 (CV):    {grid.best_score_:.4f}")
    
    y_pred_xgb = best_model.predict(X_test)
    
    # Step 7: Model evaluation
    print("\n[7/9] Evaluating models...")
    
    def evaluate(y_true, y_pred, name):
        return {
            'Model': name,
            'R2': r2_score(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred))
        }
    
    results = [
        evaluate(y_test, y_pred_lr, 'Linear Regression'),
        evaluate(y_test, y_pred_poly, 'Polynomial (deg=3)'),
        evaluate(y_test, y_pred_xgb, 'XGBoost')
    ]
    
    df_results = pd.DataFrame(results)
    df_results = df_results.round({'R2': 4, 'MAE': 2, 'RMSE': 2})
    print("\n" + df_results.to_string(index=False))
    df_results.to_csv(RESULTS_DIR / 'results_summary.csv', index=False)
    
    # Step 8: Plotting
    print("\n[8/9] Generating plots...")
    
    # --- Plot 1: Predicted vs Actual (XGBoost) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Only XGBoost points
    ax.scatter(y_test, y_pred_xgb, alpha=0.3, edgecolors='black', 
               linewidth=0.2, s=30, c='steelblue', label='XGBoost')
    
    # Fixed axis limits
    ax.set_xlim(-25, 75)
    ax.set_ylim(-25, 75)
    
    # Identity line y=x
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
    
    # --- Plot 2: Residuals ---
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
    
    # --- Plot 3: Blind validation ---
    if len(df_blind) > 0:
        print("  Performing blind validation...")
        df_blind = add_engineered_features(df_blind)
        for col in sensor_cols:
            df_blind[col] = 0
        
        X_blind = df_blind[feature_cols].values
        y_blind_true = df_blind['P_MPa'].values
        y_blind_pred = best_model.predict(X_blind)
        
        # Sort by distance for better visualization
        sort_idx = np.argsort(df_blind['d_m'].values)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(df_blind['d_m'].values[sort_idx], y_blind_pred[sort_idx],
                'r-', linewidth=2, alpha=0.8, label='XGBoost prediction')
        ax.scatter(df_blind['d_m'].values[sort_idx], y_blind_true[sort_idx],
                   s=15, c='blue', marker='o', alpha=0.5, label='FLAC3D (reference)')
        ax.set_xlabel('Distance from longwall face d, m')
        ax.set_ylabel('Vertical stress P, MPa')
        params = parse_filename(blind_filename)
        ax.set_title(
            f'Blind validation: coal seam thickness m = {params["m_plast_m"]:.0f} m, '
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
        print(f"  Blind validation: R2={blind_r2:.4f}, MAE={blind_mae:.2f} MPa")
    else:
        print("[!] Blind validation set is empty, plot not generated")
    
    # --- Plot 4: Feature Importance ---
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
    
    # --- Plot 5: Spatio-temporal heatmap ---
    print("  Generating spatio-temporal heatmap...")

    # Physical experiment parameters
    # Longwall face advance step: 1 m (field scale)
    STEP = 1.0  # m

    # Fixed monitoring point coordinates in the rock mass
    x_points = np.linspace(-5, 40, 90)  # 90 points along the longwall panel

    # Face advance stages
    stages = sorted(df_phys['stage'].unique())

    # Longwall face position at each stage:
    # Face starts at X=0 and advances STEP meters each stage
    # X_face(stage) = X_face_start + (stage - stage_min) * STEP
    stage_min = min(stages)
    x_face_start = 0.0  # initial face position (m), verify with data

    heatmap_data = np.zeros((len(stages), len(x_points)))

    for i, stage in enumerate(stages):
        # Current face position
        x_face = x_face_start + (stage - stage_min) * STEP
        
        for j, x_point in enumerate(x_points):
            # Distance from monitoring point to face
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
    cbar = fig.colorbar(im, ax=ax, label='Vertical stress P, MPa')
    ax.set_xlabel('Coordinate along longwall panel, m')
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
    
    # Step 9: Model serialization
    print("\n[9/9] Saving model...")
    joblib.dump(best_model, RESULTS_DIR / 'model.pkl')
    with open(RESULTS_DIR / 'feature_names.json', 'w', encoding='utf-8') as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"  Model: {RESULTS_DIR / 'model.pkl'}")
    print(f"  Features: {RESULTS_DIR / 'feature_names.json'}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    best_row = df_results[df_results['Model'] == 'XGBoost'].iloc[0]
    print(f"XGBoost: R2={best_row['R2']:.4f}, MAE={best_row['MAE']:.2f} MPa, RMSE={best_row['RMSE']:.2f} MPa")
    print(f"Hybrid dataset: {len(df_hybrid)} points")
    print(f"Plots: {RESULTS_DIR}")
    print(f"Model: {RESULTS_DIR}")
    
    # Save summary report
    with open(RESULTS_DIR / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write("HYBRID SURROGATE MODEL XGBOOST\n")
        f.write("Abutment Pressure Prediction\n")
        f.write("="*60 + "\n\n")
        f.write(f"Hybrid dataset: {len(df_hybrid)} points\n")
        f.write(f"  Physical modeling: {len(df_phys)} points\n")
        f.write(f"  Numerical modeling: {len(df_num)} points\n\n")
        f.write(f"Best hyperparameters: {grid.best_params_}\n\n")
        f.write(df_results.to_string(index=False))
        f.write("\n\nBlind validation: ")
        if len(df_blind) > 0:
            f.write(f"R2={blind_r2:.4f}, MAE={blind_mae:.2f} MPa\n")
        else:
            f.write("not performed\n")


if __name__ == "__main__":
    main()
