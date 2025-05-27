import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from typing import List, Optional, Dict, Tuple
from pathlib import Path
from scipy import stats


def aggregate_daily_sales(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactions to daily sales per SKU.
    Returns a DataFrame indexed by date, columns = product_id, values = quantity sold.
    """
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    daily = (
        transactions
        .set_index('timestamp')
        .groupby('product_id')['quantity']
        .resample('D')
        .sum()
        .unstack(fill_value=0)
    )
    return daily

def inventory_null_summary(inventory: pd.DataFrame):
    """Display null counts and percentages for each column."""
    null_count = inventory.isnull().sum()
    percent_null = 100 * null_count / len(inventory)
    summary = pd.DataFrame({'null_count': null_count, 'percent_null': percent_null})
    summary = summary.sort_values('null_count', ascending=False)
    display_dataframe_to_user("Inventory Null Summary", summary)

def plot_inventory_distributions(inventory: pd.DataFrame):
    """Plot histograms of stock_level and restock_quantity with distinct colors."""
    plt.figure(figsize=(8, 4))
    plt.hist(inventory['stock_level'].dropna(), bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Stock Level')
    plt.xlabel('Stock Level')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 4))
    plt.hist(inventory['restock_quantity'].dropna(), bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Restock Quantity')
    plt.xlabel('Restock Quantity')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_stock_dynamics(inventory: pd.DataFrame, transactions: pd.DataFrame, skus: list):
    """
    For each SKU, plot inventory level (blue) and overlay daily sales (orange) 
    on a twin y-axis for visual distinction.
    """
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    for sku in skus:
        inv_sku = inventory[inventory['product_id']==sku].copy()
        inv_sku['date'] = pd.to_datetime(inv_sku['date'])
        inv_ts = inv_sku.set_index('date')['stock_level']
        
        sales_sku = transactions[transactions['product_id']==sku].copy()
        sales_sku['date'] = sales_sku['timestamp'].dt.normalize()
        daily_sales = sales_sku.groupby('date')['quantity'].sum()
        
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(inv_ts.index, inv_ts.values, color='tab:blue', linewidth=2, label='Stock Level')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Level', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.plot(daily_sales.index, daily_sales.values, color='tab:orange', linestyle='--', linewidth=2, label='Daily Sales')
        ax2.set_ylabel('Daily Units Sold', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        fig.suptitle(f'Stock vs. Sales for SKU {sku}')
        ax1.grid(True, linestyle=':', linewidth=0.5)
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        plt.tight_layout()
        plt.show()


def plot_stockout_rates(inventory: pd.DataFrame):
    """Bar chart of top 10 SKUs by stockout rate with distinct bar color."""
    rates = inventory.groupby('product_id').apply(
        lambda df: np.mean(df['stock_level']==0)
    ).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(rates.index, rates.values, color='salmon', edgecolor='black')
    plt.xticks(rotation=90)
    plt.title('Top 10 SKUs by Stockout Rate')
    plt.xlabel('SKU'); plt.ylabel('Stockout Rate')
    plt.tight_layout()
    plt.show()

def plot_restock_patterns(inventory: pd.DataFrame):
    """Histogram of restock intervals and restock weekday frequency with color distinction."""
    restocks = inventory.dropna(subset=['restock_date']).copy()
    restocks['restock_date'] = pd.to_datetime(restocks['restock_date'])
    intervals = restocks.sort_values('restock_date').groupby('product_id')['restock_date'].diff().dt.days.dropna()
    
    plt.figure(figsize=(8, 4))
    plt.hist(intervals, bins=30, color='mediumpurple', edgecolor='black')
    plt.title('Histogram of Restock Intervals (days)')
    plt.xlabel('Days Between Restocks'); plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    weekdays = restocks['restock_date'].dt.day_name().value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(weekdays.index, weekdays.values, color='olive', edgecolor='black')
    plt.xticks(rotation=45)
    plt.title('Restock Events by Weekday')
    plt.xlabel('Weekday'); plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_inventory_sales_correlation(inventory: pd.DataFrame, transactions: pd.DataFrame):
    """
    Scatter of next-day sales vs. today's stock level (teal),
    and correlation heatmap (coolwarm).
    """
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    transactions['date'] = transactions['timestamp'].dt.normalize()
    daily_sales = transactions.groupby(['product_id','date'])['quantity'].sum().reset_index()
    df = inventory.merge(daily_sales, on=['product_id','date'], how='left').fillna(0)
    df['next_sales'] = df.groupby('product_id')['quantity'].shift(-1).fillna(0)
    
    plt.figure(figsize=(6, 4))
    plt.scatter(df['stock_level'], df['next_sales'], color='teal', alpha=0.5)
    plt.title("Next-Day Sales vs. Today's Stock Level")
    plt.xlabel('Stock Level Today'); plt.ylabel('Next-Day Units Sold')
    plt.tight_layout()
    plt.show()
    
    features = df[['stock_level','days_in_stock','restock_quantity','quantity']]
    corr = features.corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Inventory & Sales Correlation')
    plt.tight_layout()
    plt.show()

def plot_censoring_diagnostics(inventory: pd.DataFrame, transactions: pd.DataFrame):
    """
    Compare sales distributions on stockout (lightcoral) vs. in-stock (skyblue) days.
    """
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    transactions['date'] = transactions['timestamp'].dt.normalize()
    daily_sales = transactions.groupby(['product_id','date'])['quantity'].sum().reset_index()
    df = inventory.merge(daily_sales, on=['product_id','date'], how='left').fillna(0)
    
    stocked = df[df['stock_level']>0]['quantity']
    out_of_stock = df[df['stock_level']==0]['quantity']
    
    plt.figure(figsize=(6, 4))
    plt.hist(stocked, bins=30, alpha=0.7, label='In Stock', color='skyblue', edgecolor='black')
    plt.hist(out_of_stock, bins=30, alpha=0.7, label='Out of Stock', color='lightcoral', edgecolor='black')
    plt.legend()
    plt.title('Sales Distribution: In Stock vs. Out of Stock')
    plt.xlabel('Units Sold'); plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()



def validate_stock_vs_sales(transactions: pd.DataFrame, inventory: pd.DataFrame):
    """
    Validates whether stock levels greatly exceed sales levels.
    Computes stock-to-sales ratios, overall and per-SKU stockout days,
    and visualizes the distribution of stock-to-sales ratios.
    Returns summary statistics and a table of SKUs with highest stock-short days.
    """
    # 1. Daily sales aggregation
    transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
    transactions['date'] = transactions['timestamp'].dt.normalize()
    daily_sales = (
        transactions
        .groupby(['product_id','date'])['quantity']
        .sum()
        .reset_index()
    )
    
    # 2. Merge inventory with daily sales
    inventory['date'] = pd.to_datetime(inventory['date'])
    df = pd.merge(
        inventory, daily_sales,
        on=['product_id','date'], how='left'
    ).fillna({'quantity': 0})
    
    # 3. Compute stock-to-sales ratio (avoid division by zero)
    df['stock_to_sales'] = df['stock_level'] / (df['quantity'] + 1e-6)
    
    # 4. Overall summary of ratio
    ratio_summary = df['stock_to_sales'].describe()
    
    # 5. Proportion of days where stock < sales
    df['stock_short'] = df['stock_level'] < df['quantity']
    overall_stock_short_rate = df['stock_short'].mean()
    
    # 6. Per-SKU stock-short proportions
    sku_short = (
        df
        .groupby('product_id')['stock_short']
        .mean()
        .reset_index()
        .rename(columns={'stock_short':'prop_days_stock_short'})
        .sort_values('prop_days_stock_short', ascending=False)
    )
    
    # 7. Print summary
    print("=== Stock-to-Sales Ratio Summary ===")
    print(ratio_summary, "\n")
    print(f"Overall proportion of days where stock < sales: {overall_stock_short_rate:.2%}\n")
    print("Top 10 SKUs by proportion of days stock < sales:")
    print(sku_short.head(10).to_string(index=False))
    

    return ratio_summary, overall_stock_short_rate, sku_short


########################

def classify_demand_patterns(daily_sales: pd.DataFrame, p0_threshold: float = 0.5, cv2_threshold: float = 0.49) -> pd.DataFrame:
    """
    Classify each SKU's demand pattern into one of four categories:
      - 'Smooth':      low variability (CV^2 <= cv2_threshold) and frequent demand (p0 <= p0_threshold)
      - 'Intermittent': low variability (CV^2 <= cv2_threshold) and infrequent demand (p0 > p0_threshold)
      - 'Erratic':     high variability (CV^2 >  cv2_threshold) and frequent demand (p0 <= p0_threshold)
      - 'Lumpy':       high variability (CV^2 >  cv2_threshold) and infrequent demand (p0 > p0_threshold)
    
    Parameters:
    - daily_sales: DataFrame indexed by date with columns = SKUs, values = quantities sold.
    - p0_threshold: threshold for zero-demand proportion (e.g. 0.5)
    - cv2_threshold: threshold for coefficient of variation squared (e.g. 0.49)
    
    Returns a DataFrame with index = SKU, and columns: 
    - p0: proportion of zero-demand days, 
    - cv2: coefficient of variation squared of non-zero demand, 
    - classification: one of ['Smooth','Intermittent','Erratic','Lumpy']
    """
    results = []
    for sku in daily_sales.columns:
        series = daily_sales[sku].fillna(0)
        n = len(series)
        zeros = (series == 0).sum()
        p0 = zeros / n
        non_zero = series[series > 0]
        if len(non_zero) == 0:
            cv2 = np.nan
        else:
            cv2 = np.var(non_zero, ddof=0) / (np.mean(non_zero) ** 2)
        if p0 <= p0_threshold and cv2 <= cv2_threshold:
            cat = 'Smooth'
        elif p0 > p0_threshold and cv2 <= cv2_threshold:
            cat = 'Intermittent'
        elif p0 <= p0_threshold and cv2 > cv2_threshold:
            cat = 'Erratic'
        else:
            cat = 'Lumpy'
        results.append({'sku': sku, 'p0': p0, 'cv2': cv2, 'classification': cat})
    df_class = pd.DataFrame(results).set_index('sku')
    return df_class



########################


def aggregate_daily_data(transactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Merges transactions with product info and aggregates to daily SKU-level:
      - quantity: total units sold
      - price: mean selling price
      - avg_competitor_price: mean competitor price
      - is_promotion: max flag per day
      - category_id, quality_score, base_cost
    """
    df = transactions.merge(
        products[['product_id', 'category_id', 'quality_score', 'base_cost', 'avg_competitor_price']],
        on='product_id', how='left'
    ).copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    daily = (
        df
        .set_index('timestamp')
        .groupby('product_id')
        .resample('D')
        .agg({
            'quantity': 'sum',
            'price': 'mean',
            'avg_competitor_price': 'mean',
            'is_promotion': 'max',
            'category_id': 'first',
            'quality_score': 'first',
            'base_cost': 'first'
        })
        .dropna(subset=['price'])
        .reset_index()
    )
    daily['log_quantity'] = np.log1p(daily['quantity'])
    daily['log_price'] = np.log(daily['price'])
    daily['log_comp_price'] = np.log(daily['avg_competitor_price'])
    daily['weekday'] = daily['timestamp'].dt.dayofweek
    daily['month'] = daily['timestamp'].dt.month
    return daily

def plot_log_log_scatter_individual(df_daily: pd.DataFrame, skus: list):
    """Scatter log(quantity) vs log(price) for individual SKUs."""
    for sku in skus:
        subset = df_daily[df_daily['product_id'] == sku]
        plt.figure(figsize=(6,4))
        plt.scatter(subset['log_price'], subset['log_quantity'], alpha=0.6)
        plt.title(f'Log-Log Scatter for SKU {sku}')
        plt.xlabel('log(price)')
        plt.ylabel('log(quantity+1)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_log_log_global(df_daily: pd.DataFrame):
    """Scatter for all SKUs with global OLS fit line."""
    plt.figure(figsize=(6,4))
    plt.scatter(df_daily['log_price'], df_daily['log_quantity'], alpha=0.1)
    X = sm.add_constant(df_daily['log_price'])
    model = sm.OLS(df_daily['log_quantity'], X).fit()
    xp = np.linspace(df_daily['log_price'].min(), df_daily['log_price'].max(), 100)
    yp = model.predict(sm.add_constant(xp))
    plt.plot(xp, yp, color='red', linewidth=2, label=f'OLS β={model.params[1]:.2f}')
    plt.title('Log-Log Scatter All SKUs with OLS Fit')
    plt.xlabel('log(price)')
    plt.ylabel('log(quantity+1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return model

def plot_feature_correlation_and_vif(df_daily: pd.DataFrame):
    """
    Heatmap of feature correlations and VIF table,
    with pre-cleaning to remove inf/nan values.
    """
    # Select features
    feat = df_daily[['log_price', 'is_promotion', 'log_comp_price', 'weekday', 'month']].copy()
    
    # Replace infinite values and drop NaNs
    feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    feat.dropna(inplace=True)
    
    # Correlation heatmap
    plt.figure(figsize=(6,5))
    corr = feat.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.tight_layout()
    plt.show()

    # Compute VIF
    vif_data = pd.DataFrame({
        'feature': feat.columns,
        'VIF': [variance_inflation_factor(feat.values, i) for i in range(feat.shape[1])]
    })
    return vif_data

def compute_sku_elasticities(df_daily: pd.DataFrame, min_obs: int = 10):
    """Estimate beta per SKU and return DataFrame of slopes."""
    slopes = []
    for sku, grp in df_daily.groupby('product_id'):
        if grp['log_price'].nunique() > 1 and grp.shape[0] > min_obs:
            xi = sm.add_constant(grp['log_price'])
            mi = sm.OLS(grp['log_quantity'], xi).fit()
            slopes.append({'product_id': sku, 'beta': mi.params[1]})
    slopes_df = pd.DataFrame(slopes)
    return slopes_df

def plot_elasticity_by_category(slopes_df: pd.DataFrame, products: pd.DataFrame):
    """Boxplot of beta distributions by category."""
    df = slopes_df.merge(products[['product_id','category_id']], on='product_id', how='left')
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x='category_id', y='beta')
    plt.title('Elasticity Distribution by Category')
    plt.xlabel('Category ID')
    plt.ylabel('β')
    plt.tight_layout()
    plt.show()

def plot_beta_vs_attributes(slopes_df: pd.DataFrame, products: pd.DataFrame):
    """Scatter of beta vs. quality_score and base_cost."""
    df = slopes_df.merge(products[['product_id','quality_score','base_cost']], on='product_id', how='left')
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x='quality_score', y='beta', alpha=0.7)
    plt.title('β vs. Quality Score')
    plt.xlabel('Quality Score')
    plt.ylabel('β')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x='base_cost', y='beta', alpha=0.7)
    plt.title('β vs. Base Cost')
    plt.xlabel('Base Cost')
    plt.ylabel('β')
    plt.tight_layout()
    plt.show()

def plot_residual_diagnostics(model):
    """Residual vs fitted and QQ-plot."""
    resid = model.resid
    fitted = model.fittedvalues
    plt.figure(figsize=(5,4))
    plt.scatter(fitted, resid, alpha=0.4)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs. Fitted')
    plt.xlabel('Fitted')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.show()
    qqplot(resid, line='45', fit=True)
    plt.title('QQ-Plot of Residuals')
    plt.tight_layout()
    plt.show()

def temporal_elasticity(df_daily: pd.DataFrame):
    """Compute and print first vs. second year beta and rolling 3-month betas."""
    cutoff = df_daily['timestamp'].min() + pd.Timedelta(days=365)
    first = df_daily[df_daily['timestamp'] < cutoff]
    second = df_daily[df_daily['timestamp'] >= cutoff]
    def fit_beta(data):
        x = sm.add_constant(data['log_price'])
        return sm.OLS(data['log_quantity'], x).fit().params[1]
    b1, b2 = fit_beta(first), fit_beta(second)
    print(f"Elasticity β first year: {b1:.2f}, second year: {b2:.2f}")
    
    windows = []
    start, end = df_daily['timestamp'].min(), df_daily['timestamp'].max() - pd.Timedelta(days=90)
    cur = start
    while cur <= end:
        window = df_daily[(df_daily['timestamp'] >= cur) & (df_daily['timestamp'] < cur + pd.Timedelta(days=90))]
        if window.shape[0] > 20:
            windows.append({'mid': cur + pd.Timedelta(days=45), 'beta': fit_beta(window)})
        cur += pd.Timedelta(days=30)
    win_df = pd.DataFrame(windows)
    plt.figure(figsize=(6,4))
    plt.plot(win_df['mid'], win_df['beta'], marker='o')
    plt.title('Rolling 3-Month Elasticity')
    plt.xlabel('Window midpoint')
    plt.ylabel('β')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_price_vs_quantity_logbins(df, product_id, n_bins=10):
    """
    Assess constant elasticity assumption by binning price and plotting log-log price vs quantity.

    Parameters:
    - df: DataFrame with columns 'product_id', 'price', 'quantity'
    - product_id: specific product_id to filter and analyze
    - n_bins: number of price bins to segment
    """
    product_df = df[df['product_id'] == product_id].copy()
    product_df = product_df[(product_df['price'] > 0) & (product_df['quantity'] > 0)]

    # Bin price into quantile bins
    product_df['price_bin'] = pd.qcut(product_df['price'], q=n_bins, duplicates='drop')
    summary_df = product_df.groupby('price_bin').agg({
        'price': 'mean',
        'quantity': 'mean',
        'timestamp': 'count'
    }).rename(columns={'timestamp': 'n_obs'}).reset_index()

    # Drop bins with too few observations
    summary_df = summary_df[summary_df['n_obs'] > 5]

    # Plot log-log trend
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=summary_df,
        x=np.log(summary_df['price']),
        y=np.log(summary_df['quantity']),
        size=summary_df['n_obs'],
        legend=False
    )
    sns.regplot(
        data=summary_df,
        x=np.log(summary_df['price']),
        y=np.log(summary_df['quantity']),
        scatter=False,
        color='red',
        label='Linear Fit'
    )
    plt.title(f'Log-Log Price vs Quantity (Binned) for Product {product_id}')
    plt.xlabel('log(Price)')
    plt.ylabel('log(Quantity)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

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

class DemandAnalyzer:
    """
    Comprehensive analysis of demand patterns, price elasticity, and data quality
    for price optimization and demand forecasting.
    """
    def __init__(
        self,
        transactions: pd.DataFrame,
        inventory: pd.DataFrame,
        products: pd.DataFrame,
        figure_dir: Optional[Path] = None
    ):
        """
        Initialize with required dataframes.
        
        Args:
            transactions: DataFrame with columns [timestamp, product_id, quantity, price]
            inventory: DataFrame with columns [date, product_id, stock_level]
            products: DataFrame with product metadata
            figure_dir: Directory to save figures (optional)
        """
        self.transactions = transactions.copy()
        self.inventory = inventory.copy()
        self.products = products.copy()
        self.figure_dir = figure_dir
        
        # Ensure datetime and convert to date for consistency
        self.transactions['timestamp'] = pd.to_datetime(self.transactions['timestamp'])
        self.transactions['date'] = self.transactions['timestamp'].dt.date
        self.inventory['date'] = pd.to_datetime(self.inventory['date']).dt.date
        
        # Create daily and weekly aggregations
        self._prepare_time_aggregations()
        
        # Create figure directory if it doesn't exist
        if self.figure_dir:
            self.figure_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_time_aggregations(self):
        """Prepare daily and weekly aggregated views of the data."""
        # Daily sales
        self.daily_sales = (
            self.transactions
            .groupby(['product_id', 'date'])
            .agg({
                'quantity': 'sum',
                'price': 'mean',
                'timestamp': 'count'
            })
            .rename(columns={'timestamp': 'n_transactions'})
            .reset_index()
        )
        
        # Weekly sales
        self.weekly_sales = (
            self.transactions
            .assign(week=lambda x: x['timestamp'].dt.to_period('W'))
            .groupby(['product_id', 'week'])
            .agg({
                'quantity': 'sum',
                'price': 'mean',
                'timestamp': 'count'
            })
            .rename(columns={'timestamp': 'n_transactions'})
            .reset_index()
        )
    
    def _save_figure(self, name: str):
        """Helper method to save figures if figure_dir is set."""
        if self.figure_dir:
            plt.savefig(self.figure_dir / f"{name}.png", bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_demand_over_time(self, top_n_products: int = 5, rolling_window: int = 7):
        """Create comprehensive demand over time visualizations."""
        # Get top products by total demand
        top_products = (
            self.daily_sales
            .groupby('product_id')['quantity']
            .sum()
            .sort_values(ascending=False)
            .head(top_n_products)
            .index
            .tolist()
        )
        
        # 1. Overall demand trend
        plt.figure(figsize=(15, 6))
        total_daily = self.daily_sales.groupby('date')['quantity'].sum()
        plt.plot(total_daily.index, total_daily.values, alpha=0.5, label='Daily')
        plt.plot(total_daily.index, total_daily.rolling(rolling_window).mean(), 
                linewidth=2, label=f'{rolling_window}-day Moving Avg')
        plt.title('Overall Daily Demand')
        plt.xlabel('Date')
        plt.ylabel('Total Quantity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_figure('demand_overall_daily')

        # 2. Weekly seasonality
        plt.figure(figsize=(10, 5))
        weekday_demand = self.daily_sales.copy()
        weekday_demand['weekday'] = pd.to_datetime(weekday_demand['date']).dt.day_name()
        weekday_avg = weekday_demand.groupby('weekday')['quantity'].mean()
        # Reorder days
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_avg = weekday_avg.reindex(days_order)
        plt.bar(weekday_avg.index, weekday_avg.values)
        plt.title('Average Daily Demand by Weekday')
        plt.xlabel('Weekday')
        plt.ylabel('Average Quantity')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save_figure('demand_weekly_pattern')

        # 3. Monthly trend
        plt.figure(figsize=(15, 6))
        monthly_demand = (
            self.daily_sales
            .assign(month=pd.to_datetime(self.daily_sales['date']).dt.to_period('M'))
            .groupby('month')['quantity']
            .sum()
        )
        plt.plot(range(len(monthly_demand)), monthly_demand.values, marker='o')
        plt.title('Monthly Demand Trend')
        plt.xlabel('Month')
        plt.ylabel('Total Quantity')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(monthly_demand)), [str(x) for x in monthly_demand.index], rotation=45)
        plt.tight_layout()
        self._save_figure('demand_monthly_trend')

        # 4. Individual product trends
        plt.figure(figsize=(15, 8))
        for pid in top_products:
            prod_data = self.daily_sales[self.daily_sales['product_id'] == pid]
            plt.plot(prod_data['date'], 
                    prod_data['quantity'].rolling(rolling_window).mean(),
                    label=f'Product {pid}',
                    alpha=0.7)
        plt.title(f'Demand Trends for Top {len(top_products)} Products')
        plt.xlabel('Date')
        plt.ylabel(f'Quantity ({rolling_window}-day Moving Avg)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        self._save_figure('demand_top_products')

        # 5. Heatmap of daily demand
        plt.figure(figsize=(15, 6))
        pivot_data = (
            self.daily_sales
            .assign(
                weekday=pd.to_datetime(self.daily_sales['date']).dt.day_name(),
                week=pd.to_datetime(self.daily_sales['date']).dt.isocalendar().week
            )
            .groupby(['weekday', 'week'])['quantity']
            .sum()
            .unstack()
        )
        # Reorder weekdays
        pivot_data = pivot_data.reindex(days_order)
        sns.heatmap(pivot_data, cmap='YlOrRd', cbar_kws={'label': 'Total Quantity'})
        plt.title('Weekly Demand Patterns')
        plt.xlabel('Week of Year')
        plt.ylabel('Weekday')
        plt.tight_layout()
        self._save_figure('demand_weekly_heatmap')

    def analyze_price_demand_monotonicity(self, min_observations: int = 30, price_bins: int = 10):
        """Analyze and visualize price-demand relationships."""
        # Get products with sufficient observations
        product_counts = self.daily_sales.groupby('product_id').size()
        valid_products = product_counts[product_counts >= min_observations].index
        
        # Initialize lists for storing results
        monotonic_products = []
        non_monotonic_products = []
        
        for pid in valid_products:
            # Get product data
            prod_data = self.daily_sales[self.daily_sales['product_id'] == pid].copy()
            
            # Create price bins and calculate mean demand per bin
            prod_data['price_bin'] = pd.qcut(prod_data['price'], price_bins, duplicates='drop')
            binned_demand = prod_data.groupby('price_bin').agg({
                'price': 'mean',
                'quantity': 'mean',
                'date': 'count'
            }).rename(columns={'date': 'n_obs'})
            
            # Calculate log values
            binned_demand['log_price'] = np.log(binned_demand['price'])
            binned_demand['log_quantity'] = np.log(binned_demand['quantity'] + 1)  # Add 1 to handle zeros
            
            # Check if relationship is monotonically decreasing
            is_monotonic = (binned_demand['log_quantity'].diff() <= 0).all()
            
            if is_monotonic:
                monotonic_products.append(pid)
            else:
                non_monotonic_products.append(pid)
        
        # Plot examples
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Price-Demand Relationships in Log-Log Space', fontsize=14)
        
        # Plot examples
        for i, ax in enumerate(axes[0]):
            if i < len(monotonic_products):
                pid = monotonic_products[i]
                self._plot_log_log_relationship(pid, ax, price_bins, 
                                             title=f'Monotonic Example: Product {pid}')
        
        for i, ax in enumerate(axes[1]):
            if i < len(non_monotonic_products):
                pid = non_monotonic_products[i]
                self._plot_log_log_relationship(pid, ax, price_bins, 
                                             title=f'Non-Monotonic Example: Product {pid}')
        
        plt.tight_layout()
        self._save_figure('price_demand_examples')

        # Theoretical relationship
        plt.figure(figsize=(8, 6))
        log_prices = np.linspace(0, 2, 100)
        elasticity = -1.5
        log_demand = -elasticity * log_prices + 5
        
        plt.plot(log_prices, log_demand, 'r--', label='Theoretical Monotonic Relationship')
        plt.fill_between(log_prices, log_demand, log_demand - 0.5, 
                        color='green', alpha=0.1, label='Acceptable Range')
        plt.fill_between(log_prices, log_demand, log_demand + 0.5, 
                        color='green', alpha=0.1)
        
        plt.title('Theoretical Monotonically Decreasing Demand in Log-Log Space')
        plt.xlabel('Log(Price)')
        plt.ylabel('Log(Quantity + 1)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.text(0.5, log_demand.mean(), 
                'Elasticity = -1.5\nAs price increases,\ndemand decreases\nproportionally',
                bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()
        self._save_figure('price_demand_theoretical')

    def _plot_log_log_relationship(self, product_id: str, ax, price_bins: int, title: str):
        """Helper function to plot log-log relationship for a single product."""
        prod_data = self.daily_sales[self.daily_sales['product_id'] == product_id].copy()
        
        # Create price bins
        prod_data['price_bin'] = pd.qcut(prod_data['price'], price_bins, duplicates='drop')
        binned_demand = prod_data.groupby('price_bin').agg({
            'price': 'mean',
            'quantity': 'mean',
            'date': 'count'
        }).rename(columns={'date': 'n_obs'})
        
        # Calculate log values
        binned_demand['log_price'] = np.log(binned_demand['price'])
        binned_demand['log_quantity'] = np.log(binned_demand['quantity'] + 1)
        
        # Scatter plot of raw data (with transparency)
        ax.scatter(np.log(prod_data['price']), 
                  np.log(prod_data['quantity'] + 1),
                  alpha=0.1, color='gray', label='Raw Data')
        
        # Plot binned relationship
        ax.plot(binned_demand['log_price'], binned_demand['log_quantity'],
                'ro-', label='Binned Average', linewidth=2)
        
        # Add best fit line
        X = sm.add_constant(binned_demand['log_price'])
        model = sm.OLS(binned_demand['log_quantity'], X).fit()
        ax.plot(binned_demand['log_price'], 
                model.predict(X), 
                'b--', 
                label=f'Elasticity={model.params[1]:.2f}')
        
        ax.set_title(title)
        ax.set_xlabel('Log(Price)')
        ax.set_ylabel('Log(Quantity + 1)')
        ax.grid(True, alpha=0.3)
        ax.legend()

    def analyze_demand_patterns(self, min_periods: int = 30) -> pd.DataFrame:
        """
        Analyze demand patterns for each product.
        
        Returns DataFrame with metrics:
        - zero_demand_pct: Percentage of periods with zero demand
        - cv_squared: Coefficient of variation squared for non-zero demand
        - demand_pattern: Classification (Smooth/Intermittent/Erratic/Lumpy)
        - mean_inter_demand_interval: Average days between demands
        """
        results = []
        
        for pid in self.daily_sales['product_id'].unique():
            prod_sales = self.daily_sales[self.daily_sales['product_id'] == pid]
            
            if len(prod_sales) < min_periods:
                continue
                
            # Calculate key metrics
            zero_periods = (prod_sales['quantity'] == 0).mean()
            non_zero_qty = prod_sales[prod_sales['quantity'] > 0]['quantity']
            
            if len(non_zero_qty) > 0:
                cv_squared = (non_zero_qty.std() / non_zero_qty.mean())**2
                
                # Calculate mean inter-demand interval
                demand_dates = prod_sales[prod_sales['quantity'] > 0]['date']
                intervals = pd.Series(demand_dates).diff().dt.days
                mean_interval = intervals.mean()
                
                # Classify pattern
                if zero_periods <= 0.5:
                    if cv_squared <= 0.49:
                        pattern = 'Smooth'
                    else:
                        pattern = 'Erratic'
                else:
                    if cv_squared <= 0.49:
                        pattern = 'Intermittent'
                    else:
                        pattern = 'Lumpy'
            else:
                cv_squared = np.nan
                mean_interval = np.nan
                pattern = 'No Demand'
            
            results.append({
                'product_id': pid,
                'zero_demand_pct': zero_periods,
                'cv_squared': cv_squared,
                'mean_inter_demand_interval': mean_interval,
                'demand_pattern': pattern,
                'n_periods': len(prod_sales),
                'n_demand_periods': len(non_zero_qty)
            })
        
        return pd.DataFrame(results)

    def analyze_price_patterns(self) -> pd.DataFrame:
        """
        Analyze price patterns and variation.
        
        Returns DataFrame with metrics:
        - price_changes: Number of price changes
        - price_variation: Coefficient of variation of price
        - min/max/mean prices
        - promotion_frequency
        """
        results = []
        
        for pid in self.daily_sales['product_id'].unique():
            prod_sales = self.daily_sales[self.daily_sales['product_id'] == pid]
            
            # Analyze price changes
            price_series = prod_sales['price']
            price_changes = (price_series.diff() != 0).sum()
            
            # Basic stats
            price_stats = price_series.agg(['min', 'max', 'mean', 'std'])
            
            results.append({
                'product_id': pid,
                'n_price_changes': price_changes,
                'price_cv': price_stats['std'] / price_stats['mean'],
                'min_price': price_stats['min'],
                'max_price': price_stats['max'],
                'mean_price': price_stats['mean'],
                'price_range_pct': (price_stats['max'] - price_stats['min']) / price_stats['mean']
            })
        
        return pd.DataFrame(results)

    def analyze_stockouts(self) -> pd.DataFrame:
        """
        Analyze stockout patterns and their relationship with demand.
        
        Returns DataFrame with metrics:
        - stockout_frequency: Proportion of days with zero stock
        - demand_lost_to_stockouts: Estimated lost demand
        - avg_stock_level
        """
        results = []
        
        # Merge sales and inventory (now both have date as datetime.date type)
        daily_data = (
            self.daily_sales
            .merge(
                self.inventory[['date', 'product_id', 'stock_level']],
                on=['date', 'product_id'],
                how='left'
            )
        )
        
        for pid in daily_data['product_id'].unique():
            prod_data = daily_data[daily_data['product_id'] == pid]
            
            # Calculate stockout metrics
            stockouts = (prod_data['stock_level'] == 0).mean()
            avg_stock = prod_data['stock_level'].mean()
            
            # Compare demand during in-stock vs stockout periods
            in_stock_demand = prod_data[prod_data['stock_level'] > 0]['quantity'].mean()
            stockout_demand = prod_data[prod_data['stock_level'] == 0]['quantity'].mean()
            
            results.append({
                'product_id': pid,
                'stockout_frequency': stockouts,
                'avg_stock_level': avg_stock,
                'in_stock_demand': in_stock_demand,
                'stockout_demand': stockout_demand,
                'potential_lost_demand': max(0, in_stock_demand - stockout_demand)
            })
        
        return pd.DataFrame(results)

    def generate_summary_report(self) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive summary statistics for demand patterns,
        price patterns, and stockouts.
        """
        demand_patterns = self.analyze_demand_patterns()
        price_patterns = self.analyze_price_patterns()
        stockout_patterns = self.analyze_stockouts()
        
        # Merge all patterns
        summary = (
            demand_patterns
            .merge(price_patterns, on='product_id', how='outer')
            .merge(stockout_patterns, on='product_id', how='outer')
        )
        
        # Calculate correlations
        correlations = summary.select_dtypes(include=[np.number]).corr()
        
        # Group statistics by demand pattern
        pattern_stats = summary.groupby('demand_pattern').agg({
            'zero_demand_pct': 'mean',
            'cv_squared': 'mean',
            'price_cv': 'mean',
            'stockout_frequency': 'mean',
            'product_id': 'count'
        }).rename(columns={'product_id': 'count'})
        
        return {
            'full_summary': summary,
            'correlations': correlations,
            'pattern_stats': pattern_stats
        }

    def plot_demand_patterns(self, product_ids: Optional[List[str]] = None):
        """Plot comprehensive demand pattern visualizations."""
        if product_ids is None:
            # Select a sample of products with different patterns
            demand_patterns = self.analyze_demand_patterns()
            product_ids = []
            for pattern in demand_patterns['demand_pattern'].unique():
                pattern_prods = demand_patterns[demand_patterns['demand_pattern'] == pattern]
                if not pattern_prods.empty:
                    product_ids.append(pattern_prods.iloc[0]['product_id'])
        
        for pid in product_ids:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Demand Pattern Analysis - Product {pid}', fontsize=14)
            
            # 1. Time series of daily demand
            daily_data = self.daily_sales[self.daily_sales['product_id'] == pid]
            axes[0,0].plot(daily_data['date'], daily_data['quantity'])
            axes[0,0].set_title('Daily Demand')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Quantity')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. Histogram of non-zero demand
            non_zero = daily_data[daily_data['quantity'] > 0]['quantity']
            axes[0,1].hist(non_zero, bins=30, edgecolor='black')
            axes[0,1].set_title('Distribution of Non-Zero Demand')
            axes[0,1].set_xlabel('Quantity')
            axes[0,1].set_ylabel('Frequency')
            
            # 3. Inter-demand intervals
            demand_dates = daily_data[daily_data['quantity'] > 0]['date']
            intervals = pd.Series(demand_dates).diff().dt.days
            axes[1,0].hist(intervals.dropna(), bins=30, edgecolor='black')
            axes[1,0].set_title('Distribution of Inter-Demand Intervals')
            axes[1,0].set_xlabel('Days')
            axes[1,0].set_ylabel('Frequency')
            
            # 4. Price vs Demand scatter
            axes[1,1].scatter(daily_data['price'], daily_data['quantity'], alpha=0.5)
            axes[1,1].set_title('Price vs Demand')
            axes[1,1].set_xlabel('Price')
            axes[1,1].set_ylabel('Quantity')
            
            plt.tight_layout()
            self._save_figure(f'demand_pattern_product_{pid}')

    def plot_price_analysis(self, product_ids: Optional[List[str]] = None):
        """Plot comprehensive price analysis visualizations."""
        if product_ids is None:
            # Select products with most price variation
            price_patterns = self.analyze_price_patterns()
            product_ids = (
                price_patterns
                .nlargest(5, 'price_cv')
                ['product_id']
                .tolist()
            )
        
        for pid in product_ids:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Price Analysis - Product {pid}', fontsize=14)
            
            daily_data = self.daily_sales[self.daily_sales['product_id'] == pid]
            
            # 1. Price over time
            axes[0,0].plot(daily_data['date'], daily_data['price'])
            axes[0,0].set_title('Price Over Time')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Price')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. Price distribution
            axes[0,1].hist(daily_data['price'], bins=30, edgecolor='black')
            axes[0,1].set_title('Price Distribution')
            axes[0,1].set_xlabel('Price')
            axes[0,1].set_ylabel('Frequency')
            
            # 3. Log-Log price vs demand
            non_zero = daily_data[daily_data['quantity'] > 0]
            if not non_zero.empty:
                axes[1,0].scatter(
                    np.log(non_zero['price']),
                    np.log(non_zero['quantity']),
                    alpha=0.5
                )
                axes[1,0].set_title('Log-Log Price vs Demand')
                axes[1,0].set_xlabel('Log(Price)')
                axes[1,0].set_ylabel('Log(Quantity)')
            
            # 4. Price changes distribution
            price_changes = daily_data['price'].diff().dropna()
            axes[1,1].hist(price_changes, bins=30, edgecolor='black')
            axes[1,1].set_title('Distribution of Price Changes')
            axes[1,1].set_xlabel('Price Change')
            axes[1,1].set_ylabel('Frequency')
            
            plt.tight_layout()
            self._save_figure(f'price_analysis_product_{pid}')

    def plot_stockout_analysis(self, product_ids: Optional[List[str]] = None):
        """Plot comprehensive stockout analysis visualizations."""
        if product_ids is None:
            # Select products with highest stockout frequency
            stockout_patterns = self.analyze_stockouts()
            product_ids = (
                stockout_patterns
                .nlargest(5, 'stockout_frequency')
                ['product_id']
                .tolist()
            )
        
        for pid in product_ids:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Stockout Analysis - Product {pid}', fontsize=14)
            
            # Merge sales and inventory data
            prod_data = (
                self.daily_sales[self.daily_sales['product_id'] == pid]
                .merge(
                    self.inventory[['date', 'product_id', 'stock_level']],
                    on=['date', 'product_id'],
                    how='left'
                )
            )
            
            # 1. Stock level over time
            axes[0,0].plot(prod_data['date'], prod_data['stock_level'])
            axes[0,0].set_title('Stock Level Over Time')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Stock Level')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # 2. Demand distribution: In-stock vs Stockout
            in_stock = prod_data[prod_data['stock_level'] > 0]['quantity']
            stockout = prod_data[prod_data['stock_level'] == 0]['quantity']
            
            if not in_stock.empty and not stockout.empty:
                axes[0,1].hist([in_stock, stockout], label=['In Stock', 'Stockout'],
                             alpha=0.7, bins=20)
                axes[0,1].set_title('Demand Distribution by Stock Status')
                axes[0,1].set_xlabel('Quantity')
                axes[0,1].set_ylabel('Frequency')
                axes[0,1].legend()
            
            # 3. Stock level vs Demand scatter
            axes[1,0].scatter(prod_data['stock_level'], prod_data['quantity'], alpha=0.5)
            axes[1,0].set_title('Stock Level vs Demand')
            axes[1,0].set_xlabel('Stock Level')
            axes[1,0].set_ylabel('Quantity')
            
            # 4. Stock level distribution
            axes[1,1].hist(prod_data['stock_level'].dropna(), bins=30, edgecolor='black')
            axes[1,1].set_title('Stock Level Distribution')
            axes[1,1].set_xlabel('Stock Level')
            axes[1,1].set_ylabel('Frequency')
            
            plt.tight_layout()
            self._save_figure(f'stockout_analysis_product_{pid}')

    def plot_subcategory_price_quantity(self, min_observations: int = 30):
        """
        Plot average quantity vs average price at subcategory level.
        
        Args:
            min_observations: Minimum number of observations required for a subcategory
        """
        # Merge with products to get subcategory information
        df_with_subcat = (
            self.daily_sales
            .merge(
                self.products[['product_id', 'subcategory_identifier', 'subcategory_name']],
                on='product_id',
                how='left'
            )
        )
        
        # Calculate averages by subcategory
        subcat_stats = (
            df_with_subcat
            .groupby('subcategory_identifier')
            .agg({
                'price': ['mean', 'std', 'count'],
                'quantity': ['mean', 'std'],
                'subcategory_name': 'first'
            })
            .reset_index()
        )
        
        # Flatten column names
        subcat_stats.columns = [
            'subcategory_identifier',
            'price_mean', 'price_std', 'n_obs',
            'quantity_mean', 'quantity_std',
            'subcategory_name'
        ]
        
        # Filter subcategories with sufficient observations
        subcat_stats = subcat_stats[subcat_stats['n_obs'] >= min_observations]
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Main scatter plot
        scatter = plt.scatter(
            subcat_stats['price_mean'],
            subcat_stats['quantity_mean'],
            s=100,  # larger points
            alpha=0.6,
            c=subcat_stats['n_obs'],  # color by number of observations
            cmap='viridis'
        )
        
        # Add error bars
        plt.errorbar(
            subcat_stats['price_mean'],
            subcat_stats['quantity_mean'],
            xerr=subcat_stats['price_std'],
            yerr=subcat_stats['quantity_std'],
            fmt='none',
            alpha=0.2,
            color='gray'
        )
        
        # Add colorbar
        plt.colorbar(scatter, label='Number of Observations')
        
        # Add labels and title
        plt.xlabel('Average Price')
        plt.ylabel('Average Quantity')
        plt.title('Subcategory-Level Price-Quantity Relationship')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add annotations for top subcategories by volume
        top_n = 5
        top_subcats = subcat_stats.nlargest(top_n, 'quantity_mean')
        for _, row in top_subcats.iterrows():
            plt.annotate(
                row['subcategory_name'],
                (row['price_mean'], row['quantity_mean']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
        
        plt.tight_layout()
        self._save_figure('subcategory_price_quantity')
        
        # Create log-log version
        plt.figure(figsize=(12, 8))
        
        # Main scatter plot (log-log)
        scatter = plt.scatter(
            np.log(subcat_stats['price_mean']),
            np.log(subcat_stats['quantity_mean']),
            s=100,
            alpha=0.6,
            c=subcat_stats['n_obs'],
            cmap='viridis'
        )
        
        # Add colorbar
        plt.colorbar(scatter, label='Number of Observations')
        
        # Add labels and title
        plt.xlabel('Log(Average Price)')
        plt.ylabel('Log(Average Quantity)')
        plt.title('Subcategory-Level Price-Quantity Relationship (Log-Log)')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add annotations for top subcategories
        for _, row in top_subcats.iterrows():
            plt.annotate(
                row['subcategory_name'],
                (np.log(row['price_mean']), np.log(row['quantity_mean'])),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
        
        # Add trend line
        x = np.log(subcat_stats['price_mean'])
        y = np.log(subcat_stats['quantity_mean'])
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8, 
                label=f'Trend Line (slope={z[0]:.2f})')
        plt.legend()
        
        plt.tight_layout()
        self._save_figure('subcategory_price_quantity_log')
        
        # Print summary statistics
        print("\n=== Subcategory Price-Quantity Analysis ===")
        print(f"Number of subcategories analyzed: {len(subcat_stats)}")
        print("\nTop 5 subcategories by average quantity:")
        print(top_subcats[['subcategory_name', 'quantity_mean', 'price_mean', 'n_obs']].to_string(index=False))
        
        # Calculate price-quantity correlation
        correlation = np.corrcoef(subcat_stats['price_mean'], subcat_stats['quantity_mean'])[0,1]
        log_correlation = np.corrcoef(np.log(subcat_stats['price_mean']), 
                                    np.log(subcat_stats['quantity_mean']))[0,1]
        
        print(f"\nPrice-Quantity Correlation:")
        print(f"Raw correlation: {correlation:.3f}")
        print(f"Log-Log correlation: {log_correlation:.3f}")
        
        return subcat_stats

def run_eda(
    transactions: pd.DataFrame,
    inventory: pd.DataFrame,
    products: pd.DataFrame,
    output_dir: Optional[Path] = None,
    figure_dir: Optional[Path] = None
):
    """
    Run complete EDA pipeline and optionally save results.
    
    Args:
        transactions: Transaction data
        inventory: Inventory data
        products: Product metadata
        output_dir: Directory to save results (optional)
        figure_dir: Directory to save figures (optional)
    """
    print("\n" + "="*50)
    print("Starting Comprehensive EDA Analysis")
    print("="*50)
    
    # Create figure directory if specified
    if figure_dir:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = DemandAnalyzer(transactions, inventory, products, figure_dir=figure_dir)
    
    print("\n1. Analyzing Demand Patterns Over Time")
    print("-"*40)
    analyzer.plot_demand_over_time(top_n_products=5, rolling_window=7)
    
    print("\n2. Analyzing Price-Demand Relationships")
    print("-"*40)
    analyzer.analyze_price_demand_monotonicity(min_observations=30, price_bins=10)
    
    print("\n3. Analyzing Demand Patterns and Classifications")
    print("-"*40)
    # Generate all analyses
    summary = analyzer.generate_summary_report()
    
    # Print key findings
    print("\n=== Demand Pattern Distribution ===")
    pattern_dist = summary['full_summary']['demand_pattern'].value_counts()
    print(pattern_dist)
    
    print("\n=== Price Variation Statistics ===")
    price_stats = summary['full_summary'][['price_cv', 'n_price_changes']].describe()
    print(price_stats)
    
    print("\n=== Stockout Analysis ===")
    stockout_stats = summary['full_summary'][['stockout_frequency', 'potential_lost_demand']].describe()
    print(stockout_stats)
    
    print("\n4. Analyzing Inventory and Stock Patterns")
    print("-"*40)
    # Validate stock vs sales
    ratio_summary, overall_stock_short_rate, sku_short = validate_stock_vs_sales(
        transactions=transactions,
        inventory=inventory
    )
    
    # Plot examples of each demand pattern
    print("\n5. Plotting Detailed Pattern Examples")
    print("-"*40)
    pattern_examples = []
    for pattern in pattern_dist.index:
        example_pid = summary['full_summary'][
            summary['full_summary']['demand_pattern'] == pattern
        ].iloc[0]['product_id']
        pattern_examples.append(example_pid)
    
    analyzer.plot_demand_patterns(pattern_examples)
    analyzer.plot_price_analysis(pattern_examples)
    analyzer.plot_stockout_analysis(pattern_examples)
    
    # Print correlation analysis
    print("\n6. Correlation Analysis")
    print("-"*40)
    print("\nKey Variable Correlations:")
    key_vars = ['zero_demand_pct', 'cv_squared', 'price_cv', 'stockout_frequency']
    print(summary['correlations'].loc[key_vars, key_vars].round(3))
    
    # Save results if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary DataFrames
        for name, df in summary.items():
            df.to_csv(output_dir / f'{name}.csv')
            
        print(f"\nResults saved to {output_dir}")
    
    print("\n" + "="*50)
    print("EDA Analysis Complete")
    print("="*50)
    
    return summary 