# Hybrid Surrogate Model for Abutment Pressure Prediction using XGBoost

Hybrid surrogate machine learning model (XGBoost) for operational prediction
of abutment pressure ahead of the longface during formation of a demounting chamber.

## Key Metrics

- Hybrid dataset: 48,184 points (184 physical + 48,000 numerical)
- XGBoost R² = 0.993, MAE = 5.12 MPa
- Blind validation (H=250m, m=3m): R² = 0.9955, MAE = 4.24 MPa

## Repository Structure

```
├── model_train.py          # Model training script
├── model_predict.py        # Pressure prediction script
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore              # Excluded files
```

## Requirements

- Python 3.9+
- Dependencies: pip install -r requirements.txt

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Train (requires data files in Лаборатория/ and Нагрузка/)
python model_train.py

# Predict
python model_predict.py
```

## Input Data Format

### Physical Modeling Data (Лаборатория/coefs.xlsx)
- Column B: Distance in model (cm), scale 1:50
- Column C: Sensor ID (1-8)
- Column D: Stress concentration coefficient K
- Stage headers: "Этап N" or just number N (N = 2, 3, 5, ..., 25)

### Numerical Modeling Data (Нагрузка/*.xlsx)
- Filename format: "Nm гл Hm.xlsx" (N = seam thickness, H = depth)
- Data starts from row 8
- Each stage (1-24): X, Y, Distance [m], Sigma One [MPa]
- 4 columns per stage

## Model Features

- d_m: Distance from face (m)
- H_m: Depth (m)
- sigma_MPa: Roof strength (MPa)
- B_m: Chamber width (m)
- m_plast_m: Seam thickness (m)
- ln_d, sqrt_d, inv_d: Engineered distance features
- sensor_D1..D8: One-hot encoded sensors (physical data only)

## Target Variable

- P_MPa: Vertical abutment pressure (MPa)

## Acknowledgments

Experimental data: Saint Petersburg Mining University, Laboratory of Geomechanics.
Numerical data: FLAC3D parametric modeling.

## Authors

Nosov A.A., Anisimov K.A.

## License

MIT
