import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    mean_absolute_percentage_error,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)
from .metrics_store import (
    initialize_metrics_store,
    store_regression_metrics,
    store_classification_metrics
)

def compute_naive_error(insample: pd.Series) -> float:
    """
    Compute the mean absolute one-step naive forecast error (Q) for a series.
    Naive forecast: F_t = Y_{t-1}.  So error = Y_t - Y_{t-1}.
    """
    diffs = insample.diff().dropna()
    return diffs.abs().mean()

def compute_mase(
    insample: pd.Series,
    actual:   pd.Series,
    forecast: pd.Series
) -> float:
    """
    Compute the Mean Absolute Scaled Error (MASE) for a single series.

      MASE = mean(|Y - F|) / Q

    where Q = mean absolute naive error on the insample series.

    Args:
        insample: historical series (used only for naive error).
        actual:   true out-of-sample values.
        forecast: predicted out-of-sample values.
    """
    Q = compute_naive_error(insample)
    mae = (actual - forecast).abs().mean()
    return mae / Q if Q != 0 else np.nan

def mase_by_group(
    insample_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    group_key:   str,
    period_col:  str,
    demand_col:  str,
    forecast_col:str
) -> pd.DataFrame:
    """
    Compute MASE by group and overall.

    Args:
        insample_df: DataFrame with columns [group_key, period_col, demand_col]
                     containing your training (in-sample) data.
        forecast_df: DataFrame with columns [group_key, period_col, demand_col,
                     forecast_col] containing true vs predicted on the test set.
        group_key:   e.g. 'product_id' or 'subcategory_id'
        period_col:  e.g. 'period' or 'week'
        demand_col:  e.g. 'demand_qty'
        forecast_col:e.g. 'forecast' or your forecast column name

    Returns:
        DataFrame with columns [group_key, 'Q', 'MAE', 'mase'], plus one extra row
        with `group_key='__overall__'` giving the aggregate MASE.
    """
   
    Q = (
        insample_df
        .sort_values([group_key, period_col])
        .groupby(group_key)[demand_col]
        .apply(compute_naive_error)
        .rename('Q')
    )

    #Calculate MAE
    mae = (
        forecast_df
        .assign(_ae=lambda d: (d[demand_col] - d[forecast_col]).abs())
        .groupby(group_key)['_ae']
        .mean()
        .rename('MAE')
    )

    # 3) Combine
    df = pd.concat([Q, mae], axis=1).reset_index()
    df['mase'] = df['MAE'] / df['Q']

    # 4) Overall
    overall_mae = (forecast_df[demand_col] - forecast_df[forecast_col]).abs().mean()
    overall_Q   = compute_naive_error(
        insample_df.sort_values([group_key, period_col])[demand_col]
    )
    overall = pd.DataFrame([{
        group_key:    '__overall__',
        'Q':          overall_Q,
        'MAE':        overall_mae,
        'mase':       (overall_mae/overall_Q) if overall_Q!=0 else np.nan
    }])
    return pd.concat([df, overall], ignore_index=True)


def compute_mae(actual: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate Mean Absolute Error.
    """
    return (actual - forecast).abs().mean()

def compute_rmse(actual: pd.Series, forecast: pd.Series) -> float:
    """
    Calculate Root Mean Squared Error.
    """
    return np.sqrt(((actual - forecast) ** 2).mean())

def metrics_by_group(
    insample_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    group_key:   str,
    period_col:  str,
    demand_col:  str,
    forecast_col:str,
    seasonal_naive: bool = False
) -> pd.DataFrame:
    """
    Compute per-group and overall:
      - MAE
      - RMSE
      - MASE (if seasonal_naive=False) or sMASE (seasonal naive if True)
    """
    def _naive_error(s: pd.Series) -> float:
        if seasonal_naive:
            s_shift = s.shift(52)
        else:
            s_shift = s.shift(1)
        return (s - s_shift).abs().dropna().mean()


    Q = (
        insample_df
        .sort_values([group_key, period_col])
        .groupby(group_key)[demand_col]
        .apply(_naive_error)
        .rename('Q')
    )

    grp = forecast_df.groupby(group_key)
    MAE = grp.apply(lambda d: compute_mae(d[demand_col], d[forecast_col])).rename('MAE')
    RMSE= grp.apply(lambda d: compute_rmse(d[demand_col], d[forecast_col])).rename('RMSE')


    df = pd.concat([Q, MAE, RMSE], axis=1).reset_index()
    df['MASE'] = df['MAE'] / df['Q']

    overall_mae  = compute_mae(forecast_df[demand_col], forecast_df[forecast_col])
    overall_rmse = compute_rmse(forecast_df[demand_col], forecast_df[forecast_col])
    overall_Q    = _naive_error(
                       insample_df.sort_values([group_key, period_col])[demand_col]
                   )
    overall = pd.DataFrame([{
        group_key: '__overall__',
        'Q':       overall_Q,
        'MAE':     overall_mae,
        'RMSE':    overall_rmse,
        'MASE':    overall_mae / overall_Q if overall_Q != 0 else np.nan
    }])

    return pd.concat([df, overall], ignore_index=True)

def compute_r_squared_per_product(test_df, elasticity_results, control_cols):
    results = []

    for product_id, model_data in elasticity_results.items():
        if product_id not in test_df['product_id'].unique():
            continue

        df_test = test_df[test_df['product_id'] == product_id].copy()
        if df_test.empty:
            continue

        coefs = model_data['coef_dict']
        intercept = model_data['intercept']

        # Predict log quantity
        df_test['pred_log_quantity'] = intercept
        for feature in control_cols:
            if feature in coefs and feature in df_test.columns:
                df_test['pred_log_quantity'] += coefs[feature] * df_test[feature]

        # R^2 in log space
        r2 = r2_score(df_test['log_quantity'], df_test['pred_log_quantity'])

        results.append({
            'product_id': product_id,
            'n_obs': len(df_test),
            'R2_log_quantity': r2,
            'beta_price': model_data.get('beta_price', None)
        })

    return pd.DataFrame(results).sort_values(by='R2_log_quantity', ascending=False)

def calculate_demand_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    store_metrics: bool = True,
    model_name: Optional[str] = None,
    metrics_store: Optional[Dict] = None,
    storage_path: Optional[Path] = None
) -> Tuple[Dict[str, float], Optional[Dict]]:
    """
    Calculate call metrics
    
    """

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'r2': r2_score(y_true, y_pred),
        'bias': np.mean(y_pred - y_true),
        'relative_bias': np.mean((y_pred - y_true) / y_true) * 100
    }

    metrics.update({
        'mase': calculate_mase(y_true, y_pred),
        'smape': calculate_smape(y_true, y_pred),
        'wmape': calculate_wmape(y_true, y_pred),
        'adjusted_r2': calculate_adjusted_r2(y_true, y_pred),
        'coefficient_of_variation': np.std(y_true - y_pred) / np.mean(y_true) * 100
    })

    direction_correct = np.sum(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    total_movements = len(y_true) - 1
    metrics['directional_accuracy'] = direction_correct / total_movements * 100

    if store_metrics:
        if model_name is None or metrics_store is None:
            raise ValueError("model_name and metrics_store required when store_metrics=True")
        metrics_store = store_regression_metrics(
            metrics_store,
            model_name,
            metrics,
            storage_path=storage_path
        )
        return metrics, metrics_store
    
    return metrics, None

def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    store_metrics: bool = True,
    model_name: Optional[str] = None,
    metrics_store: Optional[Dict] = None,
    storage_path: Optional[Path] = None
) -> Tuple[Dict[str, float], Optional[Dict]]:
    """
    Calculate classification
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    

    if y_prob is not None:
        if y_prob.shape[1] == 2:  # Binary classification
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:  # Multiclass
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    

    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    

    if store_metrics:
        if model_name is None or metrics_store is None:
            raise ValueError("model_name and metrics_store required when store_metrics=True")
   
        metrics_no_cm = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
        metrics_store = store_classification_metrics(
            metrics_store,
            model_name,
            metrics_no_cm,
            storage_path=storage_path
        )
        return metrics, metrics_store
    
    return metrics, None

def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, seasonality: int = 1) -> float:
    """Calculate Mean Absolute Scaled Error."""
    n = len(y_true)
    d = np.abs(np.diff(y_true, seasonality)).mean()  # scale factor
    if d == 0:
        return np.inf
    return np.mean(np.abs(y_true - y_pred)) / d

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100

def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Mean Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def calculate_adjusted_r2(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> float:
    """Calculate Adjusted R-squared."""
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

def calculate_price_metrics(
    actual_prices: np.ndarray,
    optimal_prices: np.ndarray,
    base_costs: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate price optimization metrics.
    """
    metrics = {
        'avg_price_change': np.mean((optimal_prices - actual_prices) / actual_prices) * 100,
        'std_price_change': np.std((optimal_prices - actual_prices) / actual_prices) * 100,
        'max_price_increase': np.max((optimal_prices - actual_prices) / actual_prices) * 100,
        'max_price_decrease': np.min((optimal_prices - actual_prices) / actual_prices) * 100
    }
    
    if base_costs is not None:
        current_margins = (actual_prices - base_costs) / actual_prices * 100
        optimal_margins = (optimal_prices - base_costs) / optimal_prices * 100
        metrics.update({
            'avg_margin_current': np.mean(current_margins),
            'avg_margin_optimal': np.mean(optimal_margins),
            'margin_improvement': np.mean(optimal_margins - current_margins)
        })
    
    return metrics

def calculate_revenue_metrics(
    actual_revenue: np.ndarray,
    predicted_revenue: np.ndarray,
    by_category: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Calculate revenue impact metrics.
    """
    metrics = {
        'total_revenue_change': (np.sum(predicted_revenue) - np.sum(actual_revenue)) / np.sum(actual_revenue) * 100,
        'avg_revenue_change': np.mean((predicted_revenue - actual_revenue) / actual_revenue) * 100,
        'revenue_risk': np.std((predicted_revenue - actual_revenue) / actual_revenue) * 100,
        'pct_products_improved': np.mean(predicted_revenue > actual_revenue) * 100
    }
    
    if by_category is not None:
        category_changes = {}
        for cat in by_category.unique():
            mask = by_category == cat
            cat_change = (np.sum(predicted_revenue[mask]) - np.sum(actual_revenue[mask])) / np.sum(actual_revenue[mask]) * 100
            category_changes[cat] = cat_change
        metrics['category_changes'] = category_changes
    
    return metrics

def evaluate_backtest_results(results: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive evaluation of backtest results.
    """
    # 1. Demand Forecasting Performance
    demand_metrics = calculate_demand_metrics(
        results['demand_qty'],
        results['forecast']
    )
    
    # 2. Price Optimization Performance
    price_metrics_demand = calculate_price_metrics(
        results['price'],
        results['demand_opt_price'],
        products_df.loc[results['product_id'], 'base_cost'].values
    )
    
    price_metrics_elastic = calculate_price_metrics(
        results['price'],
        results['elastic_opt_price'],
        products_df.loc[results['product_id'], 'base_cost'].values
    )
    
    price_metrics_constrained = calculate_price_metrics(
        results['price'],
        results['constrained_opt_price'],
        products_df.loc[results['product_id'], 'base_cost'].values
    )
    
    # 3. Revenue Impact
    revenue_metrics_demand = calculate_revenue_metrics(
        results['price'] * results['demand_qty'],
        results['demand_exp_revenue'],
        results['category_id']
    )
    
    revenue_metrics_elastic = calculate_revenue_metrics(
        results['price'] * results['demand_qty'],
        results['elastic_exp_revenue'],
        results['category_id']
    )
    
    revenue_metrics_constrained = calculate_revenue_metrics(
        results['price'] * results['demand_qty'],
        results['constrained_exp_revenue'],
        results['category_id']
    )
    
    # Combine all metrics
    all_metrics = {
        'Demand Forecasting': demand_metrics,
        'Demand Model Pricing': price_metrics_demand,
        'Elasticity Model Pricing': price_metrics_elastic,
        'Constrained Model Pricing': price_metrics_constrained,
        'Demand Model Revenue': revenue_metrics_demand,
        'Elasticity Model Revenue': revenue_metrics_elastic,
        'Constrained Model Revenue': revenue_metrics_constrained
    }
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame([
        {'model': model, 'metric': metric, 'value': value}
        for model, metrics in all_metrics.items()
        for metric, value in metrics.items()
        if not isinstance(value, dict)  # Exclude nested category metrics
    ])
    
    return metrics_df

def plot_evaluation_results(metrics_df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create visualization of evaluation metrics.
    """
    # Set up the plot style
    plt.style.use('seaborn')
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    
    # 1. Demand Forecasting Metrics
    demand_metrics = metrics_df[metrics_df['model'] == 'Demand Forecasting']
    sns.barplot(data=demand_metrics, x='metric', y='value', ax=axes[0])
    axes[0].set_title('Demand Forecasting Performance')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Price Change Metrics
    price_metrics = metrics_df[metrics_df['metric'].str.contains('price|margin')]
    sns.barplot(data=price_metrics, x='metric', y='value', hue='model', ax=axes[1])
    axes[1].set_title('Price Optimization Impact')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Revenue Impact
    revenue_metrics = metrics_df[metrics_df['metric'].str.contains('revenue')]
    sns.barplot(data=revenue_metrics, x='metric', y='value', hue='model', ax=axes[2])
    axes[2].set_title('Revenue Impact Analysis')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig

def evaluate_real_world_performance(
    historical_data: pd.DataFrame,
    test_data: pd.DataFrame,
    optimization_results: pd.DataFrame
) -> pd.DataFrame:
    """
    Evaluate model performance in real-world deployment.
    
    Args:
        historical_data: Pre-optimization historical data
        test_data: Post-optimization performance data
        optimization_results: Model recommendations and predictions
        
    Returns:
        DataFrame with real-world performance metrics
    """
    metrics = []

    price_adherence = np.mean(
        np.abs(test_data['price'] - optimization_results['recommended_price']) 
        / optimization_results['recommended_price']
    ) * 100
    metrics.append({
        'category': 'Implementation',
        'metric': 'Price Recommendation Adherence %',
        'value': 100 - price_adherence
    })
    

    pre_revenue = historical_data.groupby('product_id')['revenue'].mean()
    post_revenue = test_data.groupby('product_id')['revenue'].mean()
    revenue_change = ((post_revenue - pre_revenue) / pre_revenue).mean() * 100
    metrics.append({
        'category': 'Revenue',
        'metric': 'Actual Revenue Change %',
        'value': revenue_change
    })
    

    predicted_demand = optimization_results['predicted_demand']
    actual_demand = test_data.groupby('product_id')['quantity'].sum()
    prediction_mape = np.mean(np.abs((actual_demand - predicted_demand) / actual_demand)) * 100
    metrics.append({
        'category': 'Accuracy',
        'metric': 'Production MAPE %',
        'value': prediction_mape
    })

    pre_margin = ((historical_data['price'] - historical_data['cost']) / historical_data['price']).mean() * 100
    post_margin = ((test_data['price'] - test_data['cost']) / test_data['price']).mean() * 100
    metrics.append({
        'category': 'Profitability',
        'metric': 'Margin Change %',
        'value': post_margin - pre_margin
    })
    

    pre_stock_days = historical_data.groupby('product_id')['days_in_stock'].mean()
    post_stock_days = test_data.groupby('product_id')['days_in_stock'].mean()
    inventory_impact = ((post_stock_days - pre_stock_days) / pre_stock_days).mean() * 100
    metrics.append({
        'category': 'Operations',
        'metric': 'Stock Days Change %',
        'value': inventory_impact
    })
    
    return pd.DataFrame(metrics)