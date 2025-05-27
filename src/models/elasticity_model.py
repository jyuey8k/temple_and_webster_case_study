import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Dict, Optional
import matplotlib.pyplot as plt

# We are modelling elasticity of demand with respect to price using a log-log regression model.

# We define the own price elasticity of demand as the percentage change in quantity demanded:
#     E = (dQ/Q) / (dP/P) = (dQ/dP) * (P/Q)

# If we assume linearlity in logs we can write:
#     ln(Q) = a + b * ln(P) + c * ln(P_competitor) + d * X

# From the maths this seems a bit iffy if we have intermittent demand Q since we are assuming smoothness when doing a log log regression. -> Not exactly sure how to proceed.

# Probally just use a log log regression as a baseline estimate. 

# Compute then the price from it. and compare it to price optimisation from the demand model. 

# elasticity_model.py


def fit_log_log_model(df: pd.DataFrame, groupby_col: str, control_cols: list, min_rows: int = 30, l1_ratio: float = 0.5, skip_reg = False) -> Dict[str, Dict[str, float]]:
    """
    Fit log-log ElasticNetCV regression model per subcategory to estimate price elasticity with control variables.

    Parameters:
        df (pd.DataFrame): DataFrame with log-transformed variables and control columns.
        control_cols (list): List of control feature names (can include numeric and categorical).
        min_rows (int): Minimum number of rows required to fit model for a subcategory.
        l1_ratio (float): Mix between L1 and L2 regularization (0 = Ridge, 1 = Lasso).

    Returns:
        Dictionary mapping subcategory to {'intercept', 'beta_price', 'coef_dict', 'n_obs', 'best_alpha'}.
    """
    elasticity_dict = {}

    for group, group_df in df.groupby(groupby_col):
        if len(group_df) >= min_rows:
            X = group_df[control_cols].copy()
            y = group_df['log_quantity']

            numeric_features = [col for col in control_cols if X[col].dtype in [np.float64, np.int64]]
            categorical_features = [col for col in control_cols if X[col].dtype == 'object']

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
                ]
            )

            X_prepared = preprocessor.fit_transform(X)
            model = ElasticNetCV(l1_ratio=l1_ratio, cv=5, max_iter=10000)
            model.fit(X_prepared, y)

            numeric_names = numeric_features
            categorical_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)) if categorical_features else []
            all_feature_names = numeric_names + categorical_names

            coef_dict = dict(zip(all_feature_names, model.coef_))

            beta = coef_dict.get('log_price', np.nan)

            # Fallback to OLS if beta_price is missing or fully regularized to 0
            if (np.isnan(beta) and 'log_price' in X.columns and X['log_price'].nunique() > 1) or skip_reg:
                ols_model = LinearRegression().fit(X[['log_price']], y)
                beta = ols_model.coef_[0]
                coef_dict['log_price'] = beta

            elasticity_dict[group] = {
                'intercept': model.intercept_,
                'beta_price': beta,
                'coef_dict': coef_dict,
                'n_obs': len(group_df),
                'best_alpha': model.alpha_
            }

    return elasticity_dict


def simulate_demand_curve(
    elasticity: dict,
    price_grid: np.ndarray,
    base_controls: dict = None
) -> pd.DataFrame:
    """
    Simulate demand and revenue over a range of prices using a log-log elasticity model.

    Parameters:
    ----------
    elasticity : dict
        Coefficients from fit_log_log_model() for a given product_id or subcategory.
        Must include 'intercept' and 'coef_dict'.
    price_grid : np.ndarray
        Array of prices to simulate over (e.g., np.linspace(20, 60, 20)).
    base_controls : dict, optional
        Dictionary of base values for control variables (e.g., {'is_promotion': 0, 'log_inventory': 3.5}).

    Returns:
    -------
    pd.DataFrame
        Columns: ['price', 'expected_demand', 'expected_revenue']
    """
    intercept = elasticity['intercept']
    coefs = elasticity['coef_dict']
    
    results = []
    for price in price_grid:
        log_price = np.log(price)
        
        # Start with intercept and log_price
        pred_log_demand = intercept + coefs.get('log_price', 0) * log_price

        # Add control variables
        if base_controls:
            for control, value in base_controls.items():
                pred_log_demand += coefs.get(control, 0) * value

        # Compute quantity and revenue
        demand = np.exp(pred_log_demand)
        revenue = price * demand

        results.append({
            'price': price,
            'expected_demand': demand,
            'expected_revenue': revenue
        })

    return pd.DataFrame(results)

def plot_demand_curve(
    sim_df: pd.DataFrame,
    product_id: str,
    base_price: float,
    cost_price: float = None,
    save_path: Optional[str] = None
):
    """
    Plot demand and revenue curve from simulated data.

    Args:
        sim_df (pd.DataFrame): DataFrame containing 'price', 'forecast_demand', 'expected_revenue'
        product_id (str): Product identifier
        base_price (float): Current product price
        cost_price (float, optional): Base cost to show on plot
        save_path (str, optional): Path to save the figure
    """
    if sim_df.empty:
        print(f" No simulation data for product {product_id}")
        return

    optimal_row = sim_df.loc[sim_df['expected_revenue'].idxmax()]
    optimal_price = optimal_row['price']
    optimal_revenue = optimal_row['expected_revenue']

    plt.figure(figsize=(12, 6))

    # Demand curve
    plt.subplot(1, 2, 1)
    plt.plot(sim_df['price'], sim_df['expected_demand'], marker='o')
    plt.axvline(base_price, color='gray', linestyle='--', label=f'Base Price: {base_price:.2f}')
    plt.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: {optimal_price:.2f}')
    if cost_price is not None:
        plt.axvline(cost_price, color='purple', linestyle=':', label=f'Cost Price: {cost_price:.2f}')
    plt.title(f"Demand Curve for {product_id}")
    plt.xlabel("Price")
    plt.ylabel("Forecasted Demand")
    plt.grid(True)
    plt.legend()

    # Revenue curve
    plt.subplot(1, 2, 2)
    plt.plot(sim_df['price'], sim_df['expected_revenue'], marker='o', color='green')
    plt.axvline(base_price, color='gray', linestyle='--', label=f'Base Price: {base_price:.2f}')
    plt.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price: {optimal_price:.2f}')
    if cost_price is not None:
        plt.axvline(cost_price, color='purple', linestyle=':', label=f'Cost Price: {cost_price:.2f}')
    plt.title(f"Revenue Curve for {product_id}")
    plt.xlabel("Price")
    plt.ylabel("Expected Revenue")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
