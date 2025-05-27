# Dynamic Pricing for Intermittent-Demand Products

## Overview

This project aims to design and implement a **dynamic pricing strategy for furniture and other slow-moving products** in an e-commerce setting. The work was conducted as part of a pricing optimisation case study for a Temple and Webster interview, with the objective to maximize revenue by estimating demand elasticity and selecting optimal prices subject to real-world constraints.

We explored **two complementary approaches**:

1. **Log-Log Elasticity-Based Pricing + Discrete Optimisation** 
2. **Intermittent Demand Forecasting + Discrete Optimisation** 

---

## Step 1: Log-Log Elasticity Pricing (Baseline Approach)

The initial strategy followed the traditional economic approach of estimating price elasticity via a **log-log regression model**, where:

\[
\log(\text{Demand}) = \alpha + \beta \log(\text{Price})
\]

This formulation allows us to simulate how demand responds to different prices and select the price that maximizes expected revenue:

\[
R(p) = p \cdot e^\alpha \cdot p^\beta
\]

We extended this to a **multi-variate model**, adding controls for product attributes, promotions, inventory levels, and competitor prices. For each product, we:
- Fit an ElasticNet-regularized regression model 
- Simulated revenue over a price grid
- Selected the price that maximized expected revenue
---

## Observations During EDA

However, during the exploratory data analysis (EDA), we observed that:

- **Demand was highly intermittent**, with many SKUs having long periods of zero sales
- Many products had **only one or two price points** historically, making elasticity estimation unreliable
- The **log-log model violated key assumptions**, particularly linearity in log-space and continuous demand response
- **Stockouts and visibility issues** may have influenced observed demand more than price alone

Due to these limitations, the estimated demand curves were often **flat, erratic, or driven by noise**, making price optimization based on them questionable for a large portion of the catalogue.

---

## Step 2: Intermittent Demand Forecasting + MILP Price Optimization (Extension)

To overcome these challenges, we extended our approach with a **machine learning-based two-stage forecasting model**:

- **Stage 1**: A binary classifier to predict whether demand will occur
- **Stage 2**: A regressor to estimate demand quantity, conditional on positive demand

We then **simulated demand at a range of candidate prices** for each product by modifying the price feature and re-predicting demand, resulting in a **discrete demand curve** per SKU.

Finally, we formulated a **Mixed Integer Linear Program (MILP)** using Gurobi to:

- Select one price per product from the simulated grid
- Maximize total expected revenue
- Respect inventory availability and pricing constraints

---

## Backtesting Framework

We implemented a comprehensive backtesting framework to evaluate our pricing strategies:

- **Time-Based Split**: Data is split chronologically to simulate real-world deployment
- **Multiple Models**: Tests both elasticity-based and ML-based approaches
- **Realistic Constraints**: Incorporates inventory levels and pricing rules
- **Performance Metrics**: Tracks key metrics including:
  - Demand forecasting accuracy (RMSE, MAE, MAPE)
  - Price optimization impact (% change in prices)
  - Revenue impact (expected vs actual)
  - Inventory utilization

The framework allows for:
- Comparing different pricing strategies
- Testing sensitivity to parameters
- Validating model assumptions
- Simulating real-world scenarios

---

## Metrics Storage System

We developed a robust metrics storage system to track and analyze model performance:

- **Hierarchical Storage**: Organizes metrics by:
  - Model type (regression/classification)
  - Model name
  - Timestamp
- **Metric Types**:
  - Regression metrics (RMSE, MAE, MAPE, R², etc.)
  - Classification metrics (ROC-AUC, accuracy, etc.)
  - Business metrics (revenue impact, price changes)
- **Features**:
  - JSON-based persistent storage
  - Automatic type conversion for numpy arrays
  - Historical tracking of model iterations
  - Easy retrieval and analysis of past results

---

## Results & Recommendation

By comparing both approaches:

| Approach | Strengths | Weaknesses |
|---------|-----------|-------------|
| Log-Log Elasticity | Fast, interpretable, fulfills brief | Poor fit for sparse/intermittent products |
| ML + Simulation + MILP | Handles sparse signals, better performance | Requires simulation, more complex |

We recommend combining both methods:
- Use **log-log pricing** for products with rich historical variation
- Use **ML-based simulation** for long-tail or slow-moving items

---

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure paths in `config.py`

### Running Tests
```bash
python -m pytest tests/
```

### Running Backtests
```bash
python scripts/run_backtest.py --test_months 3 --price_range 0.2 --price_steps 5
```

---

## Project Structure

```
├── data/               # Data files (not tracked in git)
├── metrics/            # Stored model metrics
├── notebooks/          # Jupyter notebooks for analysis
├── src/               
│   ├── models/        # Core model implementations
│   ├── utils/         # Utility functions
│   └── config.py      # Configuration
├── tests/             # Test suite
├── requirements.txt   # Dependencies
└── README.md
```