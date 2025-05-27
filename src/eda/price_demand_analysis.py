import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from pathlib import Path
import statsmodels.api as sm

def plot_price_demand_relationship(
    df: pd.DataFrame,
    product_ids: Optional[List[str]] = None,
    n_bins: int = 10,
    min_observations: int = 5,
    save_dir: Optional[Path] = None
):
    """
    Create comprehensive price vs demand visualizations for specified products.
    
    Args:
        df: DataFrame with columns ['product_id', 'price', 'quantity', 'timestamp']
        product_ids: List of product IDs to analyze. If None, analyze all products.
        n_bins: Number of price bins for aggregated analysis
        min_observations: Minimum observations required in a bin
        save_dir: Directory to save plots. If None, display only.
    """
    if product_ids is None:
        product_ids = df['product_id'].unique()
    
    for product_id in product_ids:
        product_df = df[df['product_id'] == product_id].copy()
        
        if len(product_df) < min_observations:
            print(f"Skipping {product_id}: insufficient data ({len(product_df)} observations)")
            continue
            
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Price-Demand Analysis for {product_id}', fontsize=14)
        
        # 1. Raw scatter plot
        axes[0,0].scatter(product_df['price'], product_df['quantity'], alpha=0.5)
        axes[0,0].set_title('Raw Price vs Quantity')
        axes[0,0].set_xlabel('Price')
        axes[0,0].set_ylabel('Quantity')
        axes[0,0].grid(True)
        
        # 2. Log-Log scatter plot
        positive_qty = product_df[product_df['quantity'] > 0]
        if not positive_qty.empty:
            axes[0,1].scatter(
                np.log(positive_qty['price']), 
                np.log(positive_qty['quantity']), 
                alpha=0.5
            )
            axes[0,1].set_title('Log-Log Price vs Quantity')
            axes[0,1].set_xlabel('Log(Price)')
            axes[0,1].set_ylabel('Log(Quantity)')
            axes[0,1].grid(True)
        
        # 3. Binned relationship
        try:
            product_df['price_bin'] = pd.qcut(product_df['price'], q=n_bins, duplicates='drop')
            summary_df = product_df.groupby('price_bin').agg({
                'price': 'mean',
                'quantity': ['mean', 'std', 'count']
            }).reset_index()
            
            summary_df.columns = ['price_bin', 'price', 'qty_mean', 'qty_std', 'n_obs']
            valid_bins = summary_df[summary_df['n_obs'] >= min_observations]
            
            if not valid_bins.empty:
                axes[1,0].errorbar(
                    valid_bins['price'],
                    valid_bins['qty_mean'],
                    yerr=valid_bins['qty_std'],
                    fmt='o-',
                    capsize=5
                )
                axes[1,0].set_title('Binned Price vs Quantity (with std dev)')
                axes[1,0].set_xlabel('Price')
                axes[1,0].set_ylabel('Mean Quantity')
                axes[1,0].grid(True)
        except Exception as e:
            print(f"Could not create binned plot for {product_id}: {str(e)}")
        
        # 4. Time series with price overlay
        ax1 = axes[1,1]
        ax2 = ax1.twinx()
        
        # Sort by timestamp for time series
        ts_df = product_df.sort_values('timestamp')
        
        # Plot quantity on primary y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Quantity', color=color1)
        ax1.plot(ts_df['timestamp'], ts_df['quantity'], color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Plot price on secondary y-axis
        color2 = 'tab:orange'
        ax2.set_ylabel('Price', color=color2)
        ax2.plot(ts_df['timestamp'], ts_df['price'], color=color2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        axes[1,1].set_title('Quantity and Price Over Time')
        
        # Rotate x-axis labels for better readability
        plt.setp(axes[1,1].xaxis.get_majorticklabels(), rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or display
        if save_dir:
            save_path = save_dir / f'price_demand_{product_id}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def analyze_price_elasticity(
    df: pd.DataFrame,
    product_ids: Optional[List[str]] = None,
    min_observations: int = 10
) -> pd.DataFrame:
    """
    Calculate price elasticity and related statistics for each product.
    
    Args:
        df: DataFrame with columns ['product_id', 'price', 'quantity']
        product_ids: List of product IDs to analyze. If None, analyze all products.
        min_observations: Minimum observations required for analysis
        
    Returns:
        DataFrame with elasticity statistics per product
    """
    if product_ids is None:
        product_ids = df['product_id'].unique()
    
    results = []
    
    for product_id in product_ids:
        product_df = df[df['product_id'] == product_id].copy()
        
        if len(product_df) < min_observations:
            continue
            
        # Calculate basic statistics
        price_stats = product_df['price'].agg(['mean', 'std', 'min', 'max'])
        qty_stats = product_df['quantity'].agg(['mean', 'std', 'min', 'max'])
        
        # Calculate correlation
        correlation = product_df['price'].corr(product_df['quantity'])
        
        # Calculate log-log elasticity for non-zero quantities
        positive_qty = product_df[product_df['quantity'] > 0]
        if len(positive_qty) >= min_observations:
            log_price = np.log(positive_qty['price'])
            log_qty = np.log(positive_qty['quantity'])
            
            # Simple OLS regression
            X = sm.add_constant(log_price)
            model = sm.OLS(log_qty, X).fit()
            elasticity = model.params[1]
            r_squared = model.rsquared
        else:
            elasticity = np.nan
            r_squared = np.nan
        
        results.append({
            'product_id': product_id,
            'n_observations': len(product_df),
            'n_nonzero_qty': len(positive_qty),
            'mean_price': price_stats['mean'],
            'price_cv': price_stats['std'] / price_stats['mean'],
            'price_range': price_stats['max'] - price_stats['min'],
            'mean_quantity': qty_stats['mean'],
            'quantity_cv': qty_stats['std'] / qty_stats['mean'],
            'price_qty_correlation': correlation,
            'log_log_elasticity': elasticity,
            'r_squared': r_squared
        })
    
    return pd.DataFrame(results)

def plot_elasticity_distribution(elasticity_df: pd.DataFrame):
    """
    Plot distribution of elasticities across products.
    
    Args:
        elasticity_df: DataFrame from analyze_price_elasticity()
    """
    plt.figure(figsize=(12, 5))
    
    # Elasticity histogram
    plt.subplot(1, 2, 1)
    plt.hist(
        elasticity_df['log_log_elasticity'].dropna(),
        bins=30,
        edgecolor='black'
    )
    plt.title('Distribution of Price Elasticities')
    plt.xlabel('Log-Log Elasticity')
    plt.ylabel('Count')
    plt.grid(True)
    
    # R-squared histogram
    plt.subplot(1, 2, 2)
    plt.hist(
        elasticity_df['r_squared'].dropna(),
        bins=30,
        edgecolor='black'
    )
    plt.title('Distribution of R-squared Values')
    plt.xlabel('R-squared')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show() 