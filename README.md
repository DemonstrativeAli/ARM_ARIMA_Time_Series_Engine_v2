<div align="center">

# 📈 ARM-ARIMA Bot API

### Adaptive Ratio Modeling with ARIMA — Automated Time-Series Forecasting Engine

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![GraphQL](https://img.shields.io/badge/GraphQL-API-E10098?style=for-the-badge&logo=graphql&logoColor=white)](https://graphql.org)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4B8BBE?style=for-the-badge)](https://www.statsmodels.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-Private-red?style=for-the-badge)](#license)

<br/>

*A production-grade time-series forecasting system that combines ARIMA modeling with Savitzky-Golay signal filtering and a custom Target Plot Generator (TPG) analytics suite — exposed through a fully-typed GraphQL API.*

</div>

---

## 🏗️ Architecture Overview

```text
                    ┌──────────────────────────────────────────────────┐
                    │              ARM-ARIMA Bot API                   │
                    │                  v0.02                           │
                    └──────────────────┬───────────────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────────────────┐
                    │            Flask Application Server              │
                    │         (app.py — Ariadne GraphQL)               │
                    ├──────────────────────────────────────────────────┤
                    │  /graphql  →  GraphiQL Explorer (GET)            │
                    │  /graphql  →  Query / Mutation Handler (POST)    │
                    │  /healthz  →  Health Check                       │
                    └────────┬─────────────────────┬───────────────────┘
                             │                     │
               ┌─────────────▼──────┐   ┌──────────▼──────────┐
               │    Query Layer     │   │   Mutation Layer     │
               │   (api/queries.py) │   │  (api/mutations.py)  │
               ├────────────────────┤   ├─────────────────────┤
               │ • resolve_arima    │   │ • upsertArimaDayByDay│
               │ • arimaFitPredict  │   │   (JSONL persistence)│
               └────────┬──────────┘   └──────────────────────┘
                        │
          ┌─────────────▼──────────────────────────────────┐
          │          ARIMA Forecasting Engine               │
          │  ┌──────────────┐  ┌─────────────────────────┐ │
          │  │ Savitzky-    │  │  Day-by-Day Prediction   │ │
          │  │ Golay Filter │→│  with Model.apply()      │ │
          │  │  (SciPy)     │  │  (statsmodels ARIMA)     │ │
          │  └──────────────┘  └─────────────────────────┘ │
          └────────────────────────────────────────────────┘
                        │
          ┌─────────────▼──────────────────────────────────┐
          │         TPG Analytics Suite                     │
          │  ┌────────────┐ ┌───────────┐ ┌──────────────┐ │
          │  │ Comparison  │ │  Scatter  │ │   Accuracy   │ │
          │  │   Plot      │ │   Plot    │ │    Plot      │ │
          │  └────────────┘ └───────────┘ └──────────────┘ │
          │  ┌────────────┐ ┌────────────────────────────┐ │
          │  │ Ratio Plot  │ │ SIR Parameters (linregress)│ │
          │  └────────────┘ └────────────────────────────┘ │
          └────────────────────────────────────────────────┘
```

---

## 🎯 Project Overview

**ARM-ARIMA Bot** is a quantitative forecasting system designed to predict financial time-series ratios using **ARIMA (AutoRegressive Integrated Moving Average)** models. The system operates on ratio-based modeling — computing `CPCP / MPN5P` ratios, applying signal preprocessing, and generating corrected predictions that outperform raw model outputs.

### Key Innovation

Rather than predicting price directly, the system models the **ratio between actual values (CPCP) and raw predicted values (MPN5P)**, then applies this learned ratio correction factor back to raw predictions — yielding significantly improved accuracy metrics:

```math
Corrected\_Prediction = MPN5P \times Predicted\_Ratio
```

This ratio-correction approach, combined with **Savitzky-Golay smoothing** and **day-by-day walk-forward validation**, achieves a **trend dispersion of ~2.3%** on the corrected predictions.

---

## 🧪 Signal Processing Pipeline

### 1. Ratio Engineering

```python
ratio = CPCP / MPN5P   # Actual / Raw_Predicted
```

The raw ratio series captures the systematic bias between actual values and base predictions. By modeling this ratio with ARIMA, we learn the correction dynamics over time.

### 2. Savitzky-Golay Filtering

Before feeding the ratio series into the ARIMA model, we apply a **Savitzky-Golay** polynomial smoothing filter from `scipy.signal`:

```python
from scipy.signal import savgol_filter

ratio_savgol = savgol_filter(ratio, window_length=7, polyorder=2)
```

| Parameter        | Value | Description                                      |
|:-----------------|:-----:|:-------------------------------------------------|
| `window_length`  |  `7`  | Number of data points in the smoothing window    |
| `polyorder`      |  `2`  | Degree of the polynomial fitted within each window |

**Why Savitzky-Golay?**

- **Preserves signal shape**: Unlike moving averages, Savgol filtering preserves higher-order moments (peaks, valleys, inflection points) of the original signal while eliminating high-frequency noise.
- **Zero phase distortion**: The filter is applied symmetrically, so there is no phase shift introduced into the smoothed data.
- **Optimal for ratio series**: The ratio series exhibits low-frequency trend dynamics with superimposed noise — Savgol precisely targets this noise without flattening the structural patterns ARIMA needs to capture.

### 3. Stationarity Testing

We use the **Augmented Dickey-Fuller (ADF)** test and **seasonal decomposition** to validate stationarity assumptions before ARIMA fitting:

```python
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# ADF Test for unit root
adfuller_test(ratio_data, title='Ratio Data')

# Multiplicative seasonal decomposition
seasonal_decompose(ratio_data, model='multiplicative', period=365)
```

ACF and PACF plots are also generated to inform the `(p, d, q)` order selection.

---

## 🔄 Day-by-Day Walk-Forward Prediction

The core forecasting strategy uses **walk-forward validation** — the model is fit once on the training set and then **applied** to progressively expanding windows of the test set using `statsmodels`' `model.apply()`:

```python
# Fit once on training data
fitted = ARIMA(y_train.values, order=(1, 0, 1)).fit()

# Walk-forward through test set
for current_date in test_dates:
    data_until_current = series[:current_date]
    applied = fitted.apply(endog=data_until_current.values)
    prediction = applied.fittedvalues[-1]  # last fitted value = prediction for current day
```

**Key Design Decisions:**

| Decision | Rationale |
|:---------|:----------|
| Single fit + `apply()` | Avoids expensive refitting at each step; uses the same parameter estimates on expanding windows |
| Date-based split | `SPLIT_DATE = 2024-12-31` — clean temporal separation prevents data leakage |
| Fitted values (not forecast) | Uses `fittedvalues[-1]` instead of `forecast(1)` — the applied model's last fitted value represents the prediction for that time step |

The system also supports an alternative **V2 resolver** (`arimaFitPredict`) that uses `forecast(steps=1)` for true out-of-sample prediction with optional exogenous variables.

---

## 📊 TPG — Target Plot Generator

The **TPG (Target Plot Generator)** is a comprehensive visualization and analytics suite that produces four diagnostic plots with embedded statistical annotations. Each plot computes and displays the **SIR parameters** (Slope, Intercept, R-value) along with **trend dispersion** and **accuracy** metrics.

### SIR Parameters Function

The core analytical engine behind TPG. Uses `scipy.stats.linregress` for OLS regression:

```python
def sir_parameters(x, y):
    analytical_params = linregress(x, y)
    
    slope     = analytical_params.slope        # Ideal: 1.0
    intercept = analytical_params.intercept     # Ideal: 0.0
    r_squared = analytical_params.rvalue ** 2   # Ideal: 1.0
    
    y_trend_line = slope * x + intercept
    dispersion = mean(|y_trend_line - y| / y_trend_line)  # Trend Dispersion %
    accuracy   = mean(|x - y| / x)                        # Relative Accuracy %
    
    return slope, intercept, r_squared, dispersion, accuracy, trend_line
```

### Four Diagnostic Plots

| # | Plot Type | Description | Key Metrics |
|:-:|:----------|:------------|:------------|
| 1 | **Comparison Plot** | Actual vs. Predicted time-series overlay | Trend slope, R², Dispersion |
| 2 | **Ratio Plot** | Predicted/Actual ratio over time with polynomial fit | Ratio stability, Trend drift |
| 3 | **Scatter Plot** | Regression scatter with `seaborn.regplot` | Slope closeness to 1.0, Intercept |
| 4 | **Accuracy Plot** | Scatter with explicit OLS trend line overlay | Point accuracy, Trend alignment |

Each plot outputs a `result` dictionary with structured metrics:

```python
result = {
    'trend_slope': float,       # OLS slope (ideal: 1.0)
    'trend_intercept': float,   # OLS intercept (ideal: 0.0)
    'trend_r2': float,          # Coefficient of determination
    'dispersion': float         # Average relative deviation from trend
}
```

---

## 🏆 Best Scores Achieved

### Day-by-Day Model (with Savitzky-Golay + ARIMA(1,0,1))

#### Raw Predictions vs. Actuals (Before ARIMA Correction)

| Metric | Value |
|:-------|------:|
| Trend Slope | `0.9872` |
| Trend Intercept | `5.1614` |
| Trend R² | `0.8821` |
| **Trend Dispersion** | **`4.51%`** |

#### Corrected Predictions vs. Actuals (After ARIMA Ratio Correction)

| Metric | Value |
|:-------|------:|
| Trend Slope | `1.0117` |
| Trend Intercept | `-3.3544` |
| Trend R² | **`0.9656`** |
| **Trend Dispersion** | **`2.35%`** |
| MAE | `7.301` |
| RMSE | `9.419` |
| **MAPE** | **`2.38%`** |

> **Result: The ARIMA ratio-correction pipeline reduces trend dispersion from ~4.5% down to ~2.3% — a 48% improvement in prediction stability.**

### One-Shot Model (auto_arima + Walk-Forward Refitting)

#### Corrected Predictions vs. Actuals

| Metric | Value |
|:-------|------:|
| Trend Slope | `1.0014` |
| Trend Intercept | `-0.2109` |
| Trend R² | **`0.9776`** |
| **Trend Dispersion** | **`2.76%`** |
| Ratio MAE | `0.027` |
| Ratio RMSE | `0.037` |
| **MAPE** | **`2.77%`** |

### Score Comparison Summary

```text
               Trend Dispersion    R²         MAPE
Raw MPN5P      ████████████  4.51%   0.8821     —
One-Shot       ██████████    2.76%   0.9776    2.77%
Day-by-Day     ████████      2.35%   0.9656    2.38%  ← Best Dispersion
```

---

## 🔌 GraphQL API

The API is built with **Ariadne** (schema-first GraphQL for Python) and served via **Flask**.

### Schema

```graphql
scalar JSON

type Query {
  arima(actual_values: JSON!, raw_predicted_values: JSON!): ArimaResponse!
  
  arimaFitPredict(
    y_train: [Float!]!
    y_test:  [Float!]!
    X_train: JSON
    X_test:  JSON
    dates_train: [String!]
    dates_test:  [String!]
    order: [Int!] = [1,0,1]
  ): ArimaResponseV2!
}

type Mutation {
  upsertArimaDayByDay(
    splitDate: String!
    order: [Int!]!
    records: [ArimaDayByDayRecordInput!]!
  ): ArimaDayByDayResponse!
}
```

### Endpoints

| Method | Path | Description |
|:------:|:-----|:------------|
| `GET` | `/graphql` | GraphiQL interactive explorer |
| `POST` | `/graphql` | GraphQL query/mutation handler |
| `GET` | `/healthz` | Health check endpoint |

### Example Query

```graphql
query RunARIMA($actual: JSON!, $rawPred: JSON!) {
  arima(actual_values: $actual, raw_predicted_values: $rawPred) {
    success
    error
    training_period {
      actual
      raw_predicted
      ratio
      corrected_predicted_values
    }
    test_period {
      actual
      raw_predicted
      ratio
      corrected_predicted_values
    }
  }
}
```

### Mutation — Persist Day-by-Day Records

```graphql
mutation Upsert($splitDate: String!, $order: [Int!]!, $records: [ArimaDayByDayRecordInput!]!) {
  upsertArimaDayByDay(splitDate: $splitDate, order: $order, records: $records) {
    success
    upsertedCount
    error
  }
}
```

Records are persisted to `arima_day_by_day_records.jsonl` for audit and replay.

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|:------|:-----------|:-------:|:--------|
| **Web Framework** | Flask | 3.0.3 | HTTP server & routing |
| **GraphQL** | Ariadne | 0.23.0 | Schema-first GraphQL implementation |
| **Time-Series** | statsmodels | 0.14.0 | ARIMA model fitting, ADF test, ACF/PACF |
| **Auto-ARIMA** | pmdarima | 2.0.4 | Automated order selection (one-shot mode) |
| **Signal Processing** | SciPy | — | Savitzky-Golay filter, `linregress` |
| **ML Metrics** | scikit-learn | 1.4.2 | MAE, RMSE, MAPE, R² |
| **Data** | pandas | 2.2.2 | DataFrame operations, date indexing |
| **Numerical** | NumPy | 1.26.4 | Array operations, vectorized math |
| **Visualization** | Matplotlib | 3.7.1 | TPG plot rendering |
| **Visualization** | Seaborn | 0.12.2 | Statistical plot styling |
| **Config** | python-dotenv | 1.0.0 | Environment variable management |

---

## 📂 Project Structure

```text
ARM_ARIMA_BOT-API_0.02/
├── app.py                          # Flask + Ariadne GraphQL server
├── run_bot.py                      # CLI bot — day-by-day prediction runner
├── arima_service.py                # Legacy service scaffolding
├── schema.graphql                  # GraphQL type definitions (SDL)
├── requirements.txt                # Python dependencies
├── tsla_data.csv                   # TSLA time-series dataset
├── arima_day_by_day_records.jsonl   # Persisted prediction records
│
├── api/
│   ├── __init__.py
│   ├── settings.py                 # Server & model configuration
│   ├── queries.py                  # GraphQL query resolvers (ARIMA engine)
│   ├── mutations.py                # GraphQL mutation resolvers (JSONL upsert)
│   ├── resolvers.py                # Legacy resolver reference
│   └── utils.py                    # SIR parameters (slope, intercept, R²)
│
└── ARIMA_Bot/
    ├── __init__.py
    ├── ARIMA_daybyday.ipynb         # Day-by-day walk-forward notebook (primary)
    ├── ARIMA_oneshot.ipynb          # One-shot auto_arima notebook
    └── tsla_data.csv               # Dataset copy for notebook execution
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ARM-ARIMA-Bot-API.git
cd ARM-ARIMA-Bot-API

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the API Server

```bash
python app.py
# Server starts at http://127.0.0.1:5000
# GraphiQL explorer available at http://127.0.0.1:5000/graphql
```

### Running the Bot (CLI)

```bash
python run_bot.py
# Loads tsla_data.csv, applies Savgol filter, runs day-by-day prediction
# Optionally sends results to the GraphQL API
```

### Configuration

Key parameters in `run_bot.py`:

```python
ARIMA_ORDER   = (1, 0, 1)    # (p, d, q) order for ARIMA
SPLIT_DATE    = "2024-12-31"  # Train/Test split boundary
USE_SAVGOL    = True          # Enable Savitzky-Golay filtering
SAVGOL_WIN    = 7             # Savgol window length
SAVGOL_POLY   = 2             # Savgol polynomial order
SEND_TO_API   = True          # Post results to GraphQL endpoint
```

---

## 📐 Mathematical Definitions

### ARIMA(p, d, q) — Model Equation

$$\phi(B)(1-B)^d X_t = \theta(B)\varepsilon_t$$

Where:
- $\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p$ (AR polynomial)
- $\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q$ (MA polynomial)
- $B$ is the backshift operator, $d$ is the differencing order

### Savitzky-Golay Filter

Fits a polynomial of degree $k$ to a window of $2m+1$ points and evaluates at the center:

$$\hat{y}_i = \sum_{j=-m}^{m} c_j \cdot y_{i+j}$$

Where $c_j$ are the convolution coefficients derived from the least-squares polynomial fit.

### Trend Dispersion

$$\text{Dispersion} = \frac{1}{n}\sum_{i=1}^{n} \frac{|\hat{y}_{trend,i} - y_i|}{|\hat{y}_{trend,i}|}$$

Measures the average relative deviation of actual values from the OLS regression trend line — our primary quality metric targeting **≤ 2.5%**.

---

## 📝 License

This is a private research project. All rights reserved.

---

<div align="center">

**Built with ❤️ for quantitative time-series analysis**

*ARIMA • Savitzky-Golay • GraphQL • Walk-Forward Validation*

</div>
