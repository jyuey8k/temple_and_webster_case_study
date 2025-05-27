import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.models.demand_forecasting import DemandForecaster
from src.models.price_optimisation import (
    generate_price_grid, 
    simulate_demand,
    optimise_prices,
    optimise_prices_independent
)
from src.models.elasticity_model import fit_log_log_model, simulate_demand_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, accuracy_score, mean_absolute_percentage_error
from src.models.metrics_store import initialize_metrics_store, store_regression_metrics, store_classification_metrics
from pathlib import Path
import src.models.evaluate as evaluate

def split_data(
    transactions_df: pd.DataFrame,
    test_months: int = 3,
    date_col: str = 'timestamp'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test sets chronologically"""
    # Convert to datetime if needed
    transactions_df = transactions_df.copy()
    transactions_df[date_col] = pd.to_datetime(transactions_df[date_col])
    
    # Find cutoff date
    max_date = transactions_df[date_col].max()
    cutoff_date = max_date - pd.DateOffset(months=test_months)
    
    # Split transactions
    train = transactions_df[transactions_df[date_col] <= cutoff_date].copy()
    test = transactions_df[transactions_df[date_col] > cutoff_date].copy()
    
    return train, test


def prepare_demand_forecaster(
    products_df: pd.DataFrame,
    train_transactions: pd.DataFrame,
    inventory_df: pd.DataFrame,
    group_key: str = 'product_id',
    date_col: str = 'timestamp'
) -> DemandForecaster:
    """Initialize and train demand forecaster on training data"""
    forecaster = DemandForecaster(
        products_df,
        train_transactions,
        inventory_df,
        group_key=group_key,
        date_col=date_col
    )
    forecaster.build_panel()
    forecaster.split()
    forecaster.fit()
    return forecaster


def prepare_elasticity_models(
    transactions_df: pd.DataFrame,
    products_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    group_key: str = 'product_id'
) -> Dict[str, dict]:
    """Fit elasticity models for each product/group"""

    # Create subcategory identifier
    products_df['subcategory'] = (
    products_df["category_id"].astype(str) + "_" + products_df["subcategory_id"].astype(str)
    )

    products_feature_list = ['product_id', 
                        'subcategory', 
                        'supplier_id',
                        'avg_competitor_price', 
                        'quality_score', 
                        'brand_id', 
                        'is_seasonal',
                        'category_id',
                        'subcategory_id',
                        ]


    # Clean Promotion Data such that if is_promottion is 0, we can assume that the product is not on promotion i.e when is_promotion = 0, promotion_type = 'No Promotion'
    transactions_df['promotion_type'] = transactions_df.apply(
    lambda x: 'No Promotion' if x['is_promotion'] == 0 else x['promotion_type'], axis=1
    )

    # Impute missing values in the quality_score column with the mean of the subcategory
    products_df['quality_score'] = products_df.groupby('subcategory')['quality_score'].transform(
    lambda x: x.fillna(round(x.mean(), 0))
    )

    # Impute missing values in the avg_competitor_price column with the mean of the subcategory
    products_df['avg_competitor_price'] = products_df.groupby('subcategory')['avg_competitor_price'].transform(
    lambda x: x.fillna(x.mean())
    )

    # Merge dataframes
    df = transactions_df.merge(products_df[products_feature_list], on='product_id', how='left')
    df = df.merge(inventory_df[['product_id', 'date', 'stock_level']], left_on=['product_id', 'timestamp'], right_on=['product_id', 'date'], how='left')

    # Extract week
    df['week'] = df['timestamp'].dt.to_period('W').dt.to_timestamp()

    # 

    df['log_price'] = np.log(df['price'])
    df['log_quantity'] = np.log(df['quantity'])
    df['log_competitor_price'] = np.log(df['avg_competitor_price'])
    df['log_inventory'] = np.log1p(df['stock_level'])

        # Aggregate by product_id and timestamp (day-level)
    agg_by_product_df = (
        df.groupby(['product_id', 'timestamp']).agg({
            'price': 'mean',
            'quantity': 'sum',
            'avg_competitor_price': 'mean',
            'quality_score': 'mean',
            'brand_id': lambda x: x.mode().iat[0],
            'supplier_id': lambda x: x.mode().iat[0],
            'is_seasonal': lambda x: x.mode().iat[0],
            'stock_level': 'mean',
            'is_promotion': lambda x: x.mode().iat[0]
        })
    )

    # Filter and compute logs
    agg_by_product_df = agg_by_product_df[agg_by_product_df['quantity'] > 0].copy()
    agg_by_product_df['log_price'] = np.log(agg_by_product_df['price'])
    agg_by_product_df['log_quantity'] = np.log(agg_by_product_df['quantity'])
    agg_by_product_df['log_competitor_price'] = np.log(agg_by_product_df['avg_competitor_price'])
    agg_by_product_df['log_inventory'] = np.log1p(agg_by_product_df['stock_level'])
    # Add control variables
    control_columns = ['quality_score', 
                       'brand_id', 
                       'is_promotion', 
                       'is_seasonal', 
                       'supplier_id', 
                       'log_price', 
                       'log_competitor_price', 
                       'log_inventory']
    
    # Fit models
    return fit_log_log_model(
        df=agg_by_product_df,
        groupby_col=group_key,
        control_cols=control_columns,
        l1_ratio=0.6
    )


def calculate_metrics(
    group_id: str,
    test_data: pd.DataFrame,
    demand_sim: pd.DataFrame,
    elasticity_sim: pd.DataFrame,
    base_price: float
) -> dict:
    """Calculate performance metrics for a single group"""
    metrics = {
        'group_id': group_id,
        'actual_price': base_price,
        'actual_revenue': test_data['price'].mean() * test_data['quantity'].sum(),
        'actual_demand': test_data['quantity'].sum(),
        'n_test_periods': len(test_data)
    }
    
    # Demand forecasting metrics
    if not demand_sim.empty:
        opt_demand_idx = demand_sim['expected_revenue'].idxmax()
        metrics.update({
            'opt_demand_price': demand_sim.loc[opt_demand_idx, 'price'],
            'opt_demand_revenue': demand_sim.loc[opt_demand_idx, 'expected_revenue'],
            'opt_demand_forecast': demand_sim.loc[opt_demand_idx, 'forecast_demand'],
            'hist_demand_revenue': demand_sim[
                demand_sim['price'].round(2) == base_price.round(2)
            ]['expected_revenue'].iloc[0]
        })
        metrics['demand_uplift_pct'] = (
            (metrics['opt_demand_revenue'] - metrics['hist_demand_revenue']) 
            / metrics['hist_demand_revenue'] * 100
        )
    
    # Elasticity metrics
    if not elasticity_sim.empty:
        opt_elastic_idx = elasticity_sim['expected_revenue'].idxmax()
        metrics.update({
            'opt_elastic_price': elasticity_sim.loc[opt_elastic_idx, 'price'],
            'opt_elastic_revenue': elasticity_sim.loc[opt_elastic_idx, 'expected_revenue'],
            'opt_elastic_demand': elasticity_sim.loc[opt_elastic_idx, 'expected_demand'],
            'hist_elastic_revenue': elasticity_sim[
                elasticity_sim['price'].round(2) == base_price.round(2)
            ]['expected_revenue'].iloc[0]
        })
        metrics['elastic_uplift_pct'] = (
            (metrics['opt_elastic_revenue'] - metrics['hist_elastic_revenue']) 
            / metrics['hist_elastic_revenue'] * 100
        )
    
    return metrics


def calculate_safe_revenue_uplift(new_revenue: np.ndarray, base_revenue: np.ndarray) -> float:
    """
    Calculate revenue uplift with safety checks for zero/small values.
    
    """
    # Filter out rows where base revenue is too small
    mask = base_revenue > 1.0  # Only consider revenue above $1
    if not mask.any():
        return 0.0
    
    uplift = ((new_revenue[mask] - base_revenue[mask]) / base_revenue[mask]).mean() * 100
    return 0.0 if np.isinf(uplift) else uplift


def run_backtest(
    products_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    test_months: int = 3,
    group_key: str = 'product_id',
    date_col: str = 'timestamp',
    features: list[str] = None,
    price_range_pct: float = 0.2,
    price_steps: int = 20,
    batch_size: int = None,
    min_clip_prob: float = None,
    metrics_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run backtest of demand forecasting and price optimization pipeline using both
    demand forecasting and elasticity models.

    """
    # Initialize metrics store
    metrics_store = initialize_metrics_store(metrics_path)
    
    # 1. Train demand forecasting model
    demand_forecaster = DemandForecaster(
        products_df=products_df,
        transactions_df=transactions_df,
        inventory_df=inventory_df,
        group_key=group_key,
        date_col=date_col
    )
    
    # Build feature panel and split data
    demand_forecaster.build_panel()
    
    # Calculate test split based on months
    total_periods = len(demand_forecaster.df_panel['period'].unique())
    test_periods = test_months * 4  # Assuming weekly frequency
    test_split = test_periods / total_periods
    
    # Split with specified features
    demand_forecaster.split(features=features, test_split=test_split)
    
    # Fit models
    demand_forecaster.fit()
    
    # Get predictions
    results = demand_forecaster.predict()
    
    # Calculate and store demand forecasting metrics
    
    if demand_forecaster.intermittent:
        test_mask = results.index.isin(demand_forecaster.df_test.index)
        test_results = results[test_mask]
        
        classification_metrics = {
            'roc_auc': roc_auc_score(
                test_results['occurrence'],
                test_results['pred_prob']
            ) if test_results['occurrence'].nunique() > 1 else None,
            'accuracy': accuracy_score(
                test_results['occurrence'],
                test_results['pred_occurrence']
            ),
            'mean_pred_prob': test_results['pred_prob'].mean(),
            'occurrence_rate': test_results['occurrence'].mean(),
            'n_test_samples': len(test_results)
        }
        metrics_store = store_classification_metrics(
            metrics_store,
            'demand_forecaster',
            classification_metrics,
            storage_path=metrics_path
        )
    
    # Regression Metrics
    test_mask = results.index.isin(demand_forecaster.df_test.index)
    test_results = results[test_mask]
    demand_mask = test_results['demand_qty'] > 0
    
    regression_metrics = {
        'rmse': np.sqrt(mean_squared_error(
            test_results['demand_qty'],
            test_results['forecast']
        )),
        'mae': mean_absolute_error(
            test_results['demand_qty'],
            test_results['forecast']
        ),
        'mape': mean_absolute_percentage_error(
            test_results[demand_mask]['demand_qty'],
            test_results[demand_mask]['forecast']
        ) if demand_mask.any() else None,
        'mean_forecast': test_results['forecast'].mean(),
        'mean_actual': test_results['demand_qty'].mean(),
        'n_test_samples': len(test_results),
        'n_demand_samples': demand_mask.sum()
    }
    metrics_store = store_regression_metrics(
        metrics_store,
        'demand_forecaster',
        regression_metrics,
        storage_path=metrics_path
    )

    train_transactions = transactions_df[
        transactions_df[date_col] <= results['period'].min()
    ]
    
    elasticity_models = prepare_elasticity_models(
        transactions_df=train_transactions, 
        products_df=products_df,
        inventory_df=inventory_df,
        group_key=group_key
    )
    

    base_prices = results.groupby(group_key)['price'].mean()
    
    # Initialize results
    opt_results = []
    
    # Process in batches if specified
    if batch_size:
        product_batches = np.array_split(base_prices.index, 
                                       len(base_prices) // batch_size + 1)
    else:
        product_batches = [base_prices.index]
    
    for batch in product_batches:
        batch_prices = base_prices[batch]
        
        # Initialize batch results
        batch_results = []
        
        for pid, base_price in tqdm(batch_prices.items(), desc="Simulating demand"):
            # Generate price grids
            demand_price_grid = generate_price_grid(
                base_price, 
                pct_range=price_range_pct,
                steps=price_steps
            )
            
            # For elasticity model, use wider range from cost price
            if pid in products_df.index:
                cost_price = products_df.loc[pid, 'base_cost']
                elastic_price_grid = np.linspace(cost_price, cost_price * 3, price_steps)
            else:
                elastic_price_grid = np.linspace(base_price * 0.5, base_price * 3, price_steps)
      
            demand_sim = simulate_demand(
                demand_forecaster,
                group_key,
                pid,
                demand_price_grid,
                clip=True,
                min_clip_prob=0.5
            )
            
         
            if pid in elasticity_models:
                elasticity_sim = simulate_demand_curve(
                    elasticity_models[pid],
                    elastic_price_grid
                )
                
                # Find optimal price from elasticity model
                if not elasticity_sim.empty:
                    opt_elastic_idx = elasticity_sim['expected_revenue'].idxmax()
                    elastic_opt = {
                        'elastic_opt_price': elasticity_sim.loc[opt_elastic_idx, 'price'],
                        'elastic_exp_demand': elasticity_sim.loc[opt_elastic_idx, 'expected_demand'],
                        'elastic_exp_revenue': elasticity_sim.loc[opt_elastic_idx, 'expected_revenue']
                    }
                else:
                    elastic_opt = {
                        'elastic_opt_price': base_price,
                        'elastic_exp_demand': None,
                        'elastic_exp_revenue': None
                    }
            else:
                elastic_opt = {
                    'elastic_opt_price': base_price,
                    'elastic_exp_demand': None,
                    'elastic_exp_revenue': None
                }
            
            # Find optimal price from demand model
            if not demand_sim.empty:
                opt_demand_idx = demand_sim['expected_revenue'].idxmax()
                demand_opt = {
                    group_key: pid,
                    'demand_opt_price': demand_sim.loc[opt_demand_idx, 'price'],
                    'demand_exp_demand': demand_sim.loc[opt_demand_idx, 'forecast_demand'],
                    'demand_exp_revenue': demand_sim.loc[opt_demand_idx, 'expected_revenue']
                }
            else:
                demand_opt = {
                    group_key: pid,
                    'demand_opt_price': None,
                    'demand_exp_demand': None,
                    'demand_exp_revenue': None
                }
            
            # Combine results
            result = {
                **demand_opt,
                **elastic_opt
            }
            batch_results.append(result)
        
        # Try constrained optimization with both models
        price_demand = {pid: simulate_demand(
            demand_forecaster, group_key, pid, 
            generate_price_grid(base_prices[pid], price_range_pct, price_steps),
            clip=True,
            min_clip_prob=0.5
        ) for pid in batch}
        
        try:
            constrained_opt = optimise_prices(price_demand, demand_forecaster.df_panel)
            
            # Update batch results with constrained optimization
            for result in batch_results:
                pid = result[group_key]
                if pid in constrained_opt:
                    opt_price = constrained_opt[pid]
                    # Get demand and revenue from price_demand dictionary
                    price_demand_df = price_demand[pid]
                    opt_row = price_demand_df[price_demand_df['price'].round(2) == round(opt_price, 2)]
                    if not opt_row.empty:
                        result.update({
                            'constrained_opt_price': opt_price,
                            'constrained_exp_demand': opt_row['forecast_demand'].iloc[0],
                            'constrained_exp_revenue': opt_row['expected_revenue'].iloc[0]
                        })
                    else:
                        # Fallback to demand model results if price not found
                        result.update({
                            'constrained_opt_price': result['demand_opt_price'],
                            'constrained_exp_demand': result['demand_exp_demand'],
                            'constrained_exp_revenue': result['demand_exp_revenue']
                        })
                else:
                    # If no constrained optimization, use demand model results
                    result.update({
                        'constrained_opt_price': result['demand_opt_price'],
                        'constrained_exp_demand': result['demand_exp_demand'],
                        'constrained_exp_revenue': result['demand_exp_revenue']
                    })
        except Exception as e:
            # Using independent optimization results as fallback
            for result in batch_results:
                # Use demand model results as fallback
                result.update({
                    'constrained_opt_price': result['demand_opt_price'],
                    'constrained_exp_demand': result['demand_exp_demand'],
                    'constrained_exp_revenue': result['demand_exp_revenue']
                })
        
        opt_results.extend(batch_results)
    
    # Convert to DataFrame
    opt_df = pd.DataFrame(opt_results)
    
    # Merge optimization results with demand forecasts
    results = results.merge(opt_df, on=group_key, how='left')
    
    # Store final optimization metrics
    base_revenue = results['price'] * results['demand_qty']
    optimization_metrics = {
        'demand_model_revenue_uplift': calculate_safe_revenue_uplift(
            results['demand_exp_revenue'],
            base_revenue
        ),
        'elasticity_model_revenue_uplift': calculate_safe_revenue_uplift(
            results['elastic_exp_revenue'],
            base_revenue
        ),
        'constrained_model_revenue_uplift': calculate_safe_revenue_uplift(
            results['constrained_exp_revenue'],
            base_revenue
        )
    }
    metrics_store = store_regression_metrics(
        metrics_store,
        'price_optimization',
        optimization_metrics,
        storage_path=metrics_path
    )
    
    return results, metrics_store


def plot_results(results: pd.DataFrame, top_n: int = 10):
    """Plot actual vs predicted demand and optimized prices from both models in chronological order"""
    # Select top products by volume
    top_products = (
        results.groupby('product_id')['demand_qty']
               .sum()
               .sort_values(ascending=False)
               .head(top_n)
               .index
    )
    
    # Plot actual vs predicted for each product
    fig, axes = plt.subplots(top_n, 2, figsize=(20, 4*top_n))
    
    # Ensure axes is 2D even with single product
    if top_n == 1:
        axes = axes.reshape(1, -1)
    
    for i, pid in enumerate(top_products):
        # Get product data and sort by period
        prod_data = results[results['product_id'] == pid].sort_values('period')
        
        # Demand plot
        ax1 = axes[i, 0]
        ax1.plot(prod_data['period'], prod_data['demand_qty'], 
                label='Actual', marker='o', linestyle='-', markersize=6)
        ax1.plot(prod_data['period'], prod_data['forecast'], 
                label='Predicted', marker='s', linestyle='-', markersize=6)
        
        # Get the first non-null value for each expected demand
        demand_exp = prod_data['demand_exp_demand'].dropna().iloc[0] if not prod_data['demand_exp_demand'].isna().all() else None
        elastic_exp = prod_data['elastic_exp_demand'].dropna().iloc[0] if not prod_data['elastic_exp_demand'].isna().all() else None
        constrained_exp = prod_data['constrained_exp_demand'].dropna().iloc[0] if not prod_data['constrained_exp_demand'].isna().all() else None
   
        # Add expected demand lines
        if demand_exp is not None:
            ax1.axhline(y=demand_exp, color='r', linestyle='--', linewidth=2,
                       label='Expected (Demand Model)')
        
        if elastic_exp is not None:
            ax1.axhline(y=elastic_exp, color='g', linestyle='--', linewidth=2,
                       label='Expected (Elasticity Model)')
        
        if constrained_exp is not None:
            ax1.axhline(y=constrained_exp, color='purple', linestyle='--', linewidth=2,
                       label='Expected (Constrained)')
        
        ax1.set_title(f'Product {pid} - Demand', pad=15)
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Demand Quantity')
        ax1.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Price plot
        ax2 = axes[i, 1]
        ax2.plot(prod_data['period'], prod_data['price'],
                color='b', label='Actual Price', marker='o', 
                linestyle='-', markersize=6, alpha=0.7)
        
        # Get the first non-null value for each optimal price
        demand_opt = prod_data['demand_opt_price'].dropna().iloc[0] if not prod_data['demand_opt_price'].isna().all() else None
        elastic_opt = prod_data['elastic_opt_price'].dropna().iloc[0] if not prod_data['elastic_opt_price'].isna().all() else None
        constrained_opt = prod_data['constrained_opt_price'].dropna().iloc[0] if not prod_data['constrained_opt_price'].isna().all() else None
        
        # Add optimal price lines
        if demand_opt is not None:
            ax2.axhline(y=demand_opt, color='r', linestyle='--', linewidth=2,
                       label='Demand Model Optimal')
        
        if elastic_opt is not None:
            ax2.axhline(y=elastic_opt, color='g', linestyle='--', linewidth=2,
                       label='Elasticity Model Optimal')
        
        if constrained_opt is not None:
            ax2.axhline(y=constrained_opt, color='purple', linestyle='--', linewidth=2,
                       label='Constrained Optimal')
        
        ax2.set_title(f'Product {pid} - Prices', pad=15)
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Price')
        ax2.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add some padding between subplots
        plt.subplots_adjust(hspace=0.5)
    
    plt.tight_layout()
    return fig


def get_summary_stats(results: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics for the backtest results from both models"""
    
    # Helper function to calculate revenue change safely
    def calc_revenue_change(predicted_revenue, actual_price, actual_qty):
        # Calculate actual revenue
        actual_revenue = actual_price * actual_qty
        
        # Filter out rows where actual revenue is too small (near zero)
        mask = actual_revenue > 1.0  # Only consider revenue above $1
        
        if not mask.any():
            return np.nan
        
        # Calculate percentage change only for valid rows
        pct_change = ((predicted_revenue[mask] - actual_revenue[mask]) / actual_revenue[mask]).mean() * 100
        
        # Return nan if the result is infinite
        return pct_change if not np.isinf(pct_change) else np.nan
    
    return pd.DataFrame({
        'metric': [
            'RMSE',
            'MAE',
            'Demand Model Avg Price Change %',
            'Elasticity Model Avg Price Change %',
            'Constrained Model Avg Price Change %',
            'Demand Model Avg Revenue Change %',
            'Elasticity Model Avg Revenue Change %',
            'Constrained Model Avg Revenue Change %'
        ],
        'value': [
            np.sqrt(mean_squared_error(
                results['demand_qty'], 
                results['forecast']
            )),
            mean_absolute_error(
                results['demand_qty'], 
                results['forecast']
            ),
            # Price changes
            ((results['demand_opt_price'] - results['price']) 
             / results['price']).mean() * 100,
            ((results['elastic_opt_price'] - results['price']) 
             / results['price']).mean() * 100,
            ((results['constrained_opt_price'] - results['price']) 
             / results['price']).mean() * 100,
            # Revenue changes with safe calculation
            calc_revenue_change(
                results['demand_exp_revenue'],
                results['price'],
                results['demand_qty']
            ),
            calc_revenue_change(
                results['elastic_exp_revenue'],
                results['price'],
                results['demand_qty']
            ),
            calc_revenue_change(
                results['constrained_exp_revenue'],
                results['price'],
                results['demand_qty']
            )
        ]
    }) 