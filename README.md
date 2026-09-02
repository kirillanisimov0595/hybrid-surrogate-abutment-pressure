# Hybrid Surrogate Model for Abutment Pressure Prediction using XGBoost

Hybrid surrogate machine learning model (XGBoost) for operational prediction
of abutment pressure ahead of the longwall face during formation of a demounting chamber.

**Update (2026):** The plotting pipeline was revised to generate **English-language figures and labels** for English-speaking users and international research use. All chart titles, axis labels, legends, and exported filenames follow English conventions.

## Key Metrics

- Hybrid dataset: 48,184 points (184 physical + 48,000 numerical)
- XGBoost R² = 0.993, MAE = 5.12 MPa
- Blind validation (H=250 m, m=3 m): R² = 0.9955, MAE = 4.24 MPa

## Repository Structure

```
├── model_train.py              # Model training + English figure export
├── model_predict.py            # Pressure prediction script
├── regenerate_figures_en.py    # Rebuild English figures without retraining
├── requirements.txt            # Python dependencies
├── docs/figures/               # Example English figures (SVG/PNG)
├── README.md                   # This file
└── .gitignore                  # Excluded files
```

## Requirements

- Python 3.9+
- Dependencies: `pip install -r requirements.txt`

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (requires data files in Лаборатория/ and Нагрузка/)
python model_train.py

# Predict
python model_predict.py

# Rebuild English figures from a saved model (fast, no retraining)
python regenerate_figures_en.py
```

## English Figures

After training (or if `results/model.pkl` already exists), run:

```bash
python regenerate_figures_en.py
```

Main outputs in `results/`:

| File | Description |
|------|-------------|
| `figure4_blind_validation.png/.svg` | Blind validation: XGBoost vs FLAC3D (H=250 m, m=3 m) |
| `figure5_spacetime_heatmap.png/.svg` | Spatio-temporal evolution of abutment pressure |
| `fig1_predicted_vs_actual.*` | Predicted vs actual (test set) |
| `fig2_residuals.*` | Residual plot |
| `fig4_feature_importance.*` | Feature importance |

Example copies are also stored in `docs/figures/`.

## Input Data Format

### Physical Modeling Data (`Лаборатория/coefs.xlsx`)
- Column B: Distance in model (cm), scale 1:50
- Column C: Sensor ID (1-8)
- Column D: Stress concentration coefficient K
- Stage headers: "Этап N" or just number N (N = 2, 3, 5, ..., 25)

### Numerical Modeling Data (`Нагрузка/*.xlsx`)
- Filename format: "Nm гл Hm.xlsx" (N = seam thickness, H = depth)
- Data starts from row 8
- Each stage (1-24): X, Y, Distance [m], Sigma One [MPa]
- 4 columns per stage

## Model Features

- `d_m`: Distance from face (m)
- `H_m`: Depth (m)
- `sigma_MPa`: Roof strength (MPa)
- `B_m`: Chamber width (m)
- `m_plast_m`: Seam thickness (m)
- `ln_d`, `sqrt_d`, `inv_d`: Engineered distance features
- `sensor_D1..D8`: One-hot encoded sensors (physical data only)

## Target Variable

- `P_MPa`: Vertical abutment pressure (MPa)

## Acknowledgments

Experimental data: Saint Petersburg Mining University, Laboratory of Geomechanics.
Numerical data: FLAC3D parametric modeling.

## Authors

Nosov A.A., Anisimov K.A.

## License

MIT
