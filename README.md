# Spanish Electricity Price Forecasting

Comparative study of tree-based machine learning models for day-ahead electricity price forecasting in the Spanish market (OMIE). The project evaluates **XGBoost**, **Random Forest**, and **M5Prime** across different feature sets, lookback windows, and feature selection strategies.

## Problem

Electricity prices in Spain are determined hourly by the OMIE market. Accurate 24-hour-ahead forecasts are valuable for energy producers, consumers, and grid operators. This project frames the problem as a **multi-step regression task**: given a sliding window of past observations, predict the next 24 hourly prices.

## Dataset

Hourly data spanning **2021–2022**, combining signals from public Spanish sources:

| Feature group | Variables | Source |
|---|---|---|
| Market | Electricity price (€/MWh) — **target** | OMIE |
| Demand & generation | Electrical demand, solar and wind generation (MWh) | REE |
| Energy costs | Natural gas price (€/MWh) | MIBGAS |
| Weather | Min/max temperature and average wind speed for Madrid, Barcelona, Sevilla, Valencia | AEMET |
| Time encoding | Hour and day-of-week as sine/cosine pairs | — |

The processed dataset (`data/electricSystem/dataset.csv`) has ~17,500 hourly rows and 22 features after preprocessing.

## Methodology

```
Raw data  →  Preprocessing  →  Feature selection  →  Sliding window  →  Model training  →  Evaluation
```

1. **Preprocessing** — Z-score normalisation, temporal train/test split, optional differencing to model residuals.
2. **Feature selection** — two strategies compared:
   - `SelectKBest` with F-regression score
   - Custom **CFS** (Correlation-based Feature Selection) using symmetrical uncertainty and k-NN mutual information estimation
3. **Sliding window** — configurable `past_history` (24 / 48 / 72 / 168 hours) and 24-hour forecast horizon. Optionally includes *future variables* (forecasted demand, gas price, weather) to simulate real market conditions.
4. **Models** — hyperparameter grid defined in `configs/parameters*.json`; experiments run via `src/main.py`.
5. **Metrics** — MAPE, MAE evaluated on denormalised predictions.

## Results

Best configuration: **Random Forest**, 168-hour lookback, 8 features (demand, generation, time encoding).

| Model | Past history | Features | MAPE | MAE (€/MWh) |
|---|---|---|---|---|
| Random Forest | 168 h | 8 | **12.55%** | 22.10 |
| Random Forest | 48 h | 9 | 13.86% | 24.92 |
| M5Prime | 48 h | 5 | 14.26% | 25.01 |
| M5Prime | 24 h | 24 h | 14.22% | 25.18 |

Full results in [`results/results_ml.csv`](results/results_ml.csv).

## Project Structure

```
├── configs/                  # Experiment configurations (JSON)
├── data/
│   ├── electricSystem/
│   │   └── dataset.csv       # Processed dataset (2021–2022, hourly)
│   └── raw/                  # Source files (REE, OMIE, MIBGAS, AEMET)
├── notebooks/
│   └── data_exploration.ipynb
├── references/               # Papers consulted
├── results/                  # Experiment results (CSV summaries)
└── src/
    ├── cfs/                  # Correlation-based Feature Selection implementation
    │   ├── CFS.py
    │   ├── mutual_information.py
    │   └── entropy_estimators.py
    ├── main.py               # Training pipeline entry point
    ├── preprocessing.py      # Windowing, normalisation, train/test split
    ├── models.py             # XGBoost, Random Forest, M5Prime wrappers
    ├── metrics.py            # MAPE, MAE, MSE evaluation
    └── report_generator.py   # LaTeX table generation from results CSVs
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
cd src
python main.py
```

Experiment parameters (date ranges, feature sets, model hyperparameters) are configured in `configs/parameters*.json`. Results are written to `results/results_ml.csv`.

To explore a different configuration, edit or copy a JSON file in `configs/` and pass its path to `main_ml()` in `main.py`.

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `pandas` · `NumPy` · `m5py` · `matplotlib`
