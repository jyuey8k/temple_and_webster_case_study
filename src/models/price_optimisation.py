import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import joblib
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
from tqdm import tqdm

def generate_price_grid(base_price: float, pct_range: float = 0.2, steps: int = 20) -> np.ndarray:
    return np.linspace(base_price * (1 - pct_range), base_price * (1 + pct_range), steps)


def simulate_demand(forecaster, group_key:str, group_id: str, price_grid: np.ndarray, clip = False, min_clip_prob = None) -> pd.DataFrame:
    df_group = forecaster.df_panel[forecaster.df_panel[group_key] == group_id]
    if df_group.empty:
        print(f" No panel data for subcategory={group_id}")
        return pd.DataFrame()

    latest_row = df_group.sort_values('period').iloc[-1].copy()

    sim_rows = []
    for price in price_grid:
        row = latest_row.copy()
        row['price'] = price
        sim_rows.append(row)
    sim_df = pd.DataFrame(sim_rows).reset_index(drop=True)
    sim_df[group_key] = group_id  # assign uniformly after DataFrame creation

    # Check feature alignment
    missing_cols = set(forecaster.X_tr_cls.columns) - set(sim_df.columns)
    if missing_cols:
        print(f" Missing columns for subcategory {group_id}: {missing_cols}")
        return pd.DataFrame()

    X_sim = sim_df[forecaster.X_tr_cls.columns]

    if clip: # The reason why we add a clip is that the demand is very small if we stick with intermittent model; in general cases a lot of products on a DAILY basis will not sell; we could add logic 
        #depending on the product category/ seasonality/ other factors to determine the minimum probability of occurrence that is lets say 0.5; if the probability of occurrence is less than 0.5, we clip the demand to 0.5 instead
        # as a base belief that the product will have a 50% chance of selling. This can be adjusted or even added as a probaibility function to the model.

        p_occurrence = forecaster.clf_search.predict_proba(X_sim)[:, 1] if forecaster.intermittent else np.ones(len(X_sim))
        p_occurrence = np.clip(p_occurrence, min_clip_prob, 1)
    else:
        p_occurrence = forecaster.clf_search.predict_proba(X_sim)[:, 1] if forecaster.intermittent else np.ones(len(X_sim))
    expected_qty = forecaster.reg_search.predict(X_sim)
    if clip:
        expected_qty = np.clip(expected_qty, 0, 1000000)
    sim_df['forecast_demand'] = p_occurrence * expected_qty
    sim_df['expected_revenue'] = sim_df['forecast_demand'] * sim_df['price']
    return sim_df[[group_key, 'price', 'forecast_demand', 'expected_revenue']]



def optimise_prices_independent(price_demand_dict: dict) -> dict:
    """Fallback: optimise prices independently by selecting the price with highest revenue."""
    selected_prices = {}
    for pid, df in price_demand_dict.items():
        best_row = df.loc[df['expected_revenue'].idxmax()]
        selected_prices[pid] = best_row['price']
    return selected_prices


def optimise_prices_batch(
    price_demand_dict: dict,
    product_metadata: pd.DataFrame,
    batch_size: int = 50,
    max_total_demand: float = None,
) -> dict:
    """
    Optimise prices across products in batches to handle large-scale problems.
    
    Args:
        price_demand_dict (dict): {product_id: df with price, forecast_demand, expected_revenue}
        product_metadata (pd.DataFrame): Product metadata with constraints
        batch_size (int): Number of products to optimize in each batch
        max_total_demand (float): Optional constraint on total demand
        
    Returns:
        dict: {product_id: selected optimal price}
    """
    # Sort products by revenue potential for prioritization
    product_revenues = {
        pid: df['expected_revenue'].max() 
        for pid, df in price_demand_dict.items()
    }
    sorted_products = sorted(
        product_revenues.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Process in batches
    selected_prices = {}
    for i in range(0, len(sorted_products), batch_size):
        batch_products = sorted_products[i:i + batch_size]
        batch_pids = [p[0] for p in batch_products]
        
        # Create batch dictionaries
        batch_price_demand = {
            pid: price_demand_dict[pid] 
            for pid in batch_pids
        }
        
        try:
            # Try optimizing the batch
            batch_prices = optimise_prices_core(
                batch_price_demand,
                product_metadata,
                max_total_demand=max_total_demand if i == 0 else None  # Only apply total demand constraint to first batch
            )
            selected_prices.update(batch_prices)
            
        except gp.GurobiError:
            # Fallback to independent optimization for failed batch
            print(f"Optimization failed for batch {i//batch_size + 1}, using independent optimization")
            batch_prices = optimise_prices_independent(batch_price_demand)
            selected_prices.update(batch_prices)
            
    return selected_prices

def optimise_prices_core(
    price_demand_dict: dict,
    product_metadata: pd.DataFrame,
    max_total_demand: float = None,
) -> dict:
    """
    The main workhorse for the optimisation. We fomrulate the problem by setting up the price
    optimisation as a linear programming problem on a graph, where the nodes are the products
    and prices and the edges are constraints. On the left hand side we have each product and the right hand side is the price
    the edges is the revenue that we are trying to maximise. We can think of the problem as eliminating all the edgers that 
    do no make sense with the constraints. 
    """
    try:
        m = gp.Model()
        m.setParam('TimeLimit', 60)  # 60 second time limit
        m.setParam('MIPGap', 0.01)   # 1% optimality gap
        m.setParam('Threads', 4)      # Limit threads
        
        x = {}  # binary decision variables
        product_ids = list(price_demand_dict.keys())
        pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}

        
        for i, pid in enumerate(product_ids):
            df = price_demand_dict[pid]
            x.update({(i, j): m.addVar(vtype=GRB.BINARY) 
                     for j in range(len(df))})

        # Set an objectie function to maximise expected revenue; basically get the product of price and demand
        obj_expr = gp.quicksum(
            price_demand_dict[product_ids[i]].iloc[j]['expected_revenue'] * x[i, j]
            for i in range(len(product_ids))
            for j in range(len(price_demand_dict[product_ids[i]]))
        )
        # Set to max the objective function
        m.setObjective(obj_expr, GRB.MAXIMIZE)

        # Add a constraint 

        # One price per product constraint 
        for i in range(len(product_ids)):
            df = price_demand_dict[product_ids[i]]
            m.addConstr(gp.quicksum(x[i, j] for j in range(len(df))) == 1)

        # Optional total demand constraint (
        if max_total_demand is not None:
            m.addConstr(
                gp.quicksum(
                    price_demand_dict[product_ids[i]].iloc[j]['forecast_demand'] * x[i, j]
                    for i in range(len(product_ids))
                    for j in range(len(price_demand_dict[product_ids[i]]))
                ) <= max_total_demand
            )

        # Stock level constraints we cant sell more than we have; in our toy example we have a capped one
        for pid in product_ids:
            if pid in product_metadata.index:
                stock = product_metadata.loc[pid, 'stock_level']
                i = pid_to_idx[pid]
                df = price_demand_dict[pid]
                m.addConstr(
                    gp.quicksum(df.iloc[j]['forecast_demand'] * x[i, j] 
                               for j in range(len(df))) <= stock
                )

        # Optimize with error handling
        m.optimize()

        # Extract results
        selected_prices = {}
        if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
            for (i, j), var in x.items():
                if var.X > 0.5:  # Binary variable is selected
                    pid = product_ids[i]
                    selected_prices[pid] = price_demand_dict[pid].iloc[j]['price']
            return selected_prices
        else:
            raise gp.GurobiError(f"Optimization failed with status {m.status}")

    except gp.GurobiError as e:
        print(f"Gurobi optimization failed: {str(e)}")
        raise e

def optimise_prices(
    price_demand_dict: dict,
    product_metadata: pd.DataFrame,
    max_total_demand: float = None,
    batch_size: int = 50
) -> dict:
    """
    Main optimization function with batching and fallback handling.
    """
    try:
        # Try batch optimization first
        return optimise_prices_batch(
            price_demand_dict,
            product_metadata,
            batch_size=batch_size,
            max_total_demand=max_total_demand
        )
    except Exception as e:
        print(f"Batch optimization failed, falling back to independent optimization: {str(e)}")
        return optimise_prices_independent(price_demand_dict)

def run_end_to_end_optimisation(
    forecaster, 
    base_prices: dict, 
    product_metadata: pd.DataFrame,
    pct_range: float = 0.2, 
    steps: int = 20, 
    group_key: str = 'product_id',
    batch_size: int = 50,
    clip: bool = True,
    min_clip_prob: float = 0.5
) -> pd.DataFrame:
    """
    Run end-to-end optimization with improved efficiency.
    """
    price_demand_dict = {}
    for pid, base_price in tqdm(base_prices.items(), desc="Simulating demand"): # Add a progres bar or we'll never know how long it will take
        grid = generate_price_grid(base_price, pct_range=pct_range, steps=steps)
        sim_df = simulate_demand(
            forecaster, 
            group_key, 
            group_id=pid, 
            price_grid=grid,
            clip=clip,
            min_clip_prob=min_clip_prob
        )
        if not sim_df.empty:
            price_demand_dict[pid] = sim_df

    optimal_prices = optimise_prices(
        price_demand_dict,
        product_metadata,
        batch_size=batch_size
    )
    
    return pd.DataFrame.from_dict(
        optimal_prices, 
        orient='index', 
        columns=['optimal_price']
    ).reset_index(names=group_key)

def save_forecaster(forecaster, filepath: str):
    joblib.dump(forecaster, filepath)

def load_forecaster(filepath: str):
    return joblib.load(filepath)

def plot_demand_curve(
    forecaster,
    group_key: str,
    group_id: str,
    base_price: float,
    pct_range: float = 0.2,
    steps: int = 20,
    clip: bool = False,
    min_clip_prob: float = None,
    save_path: Optional[str] = None,
    supress_output: bool = True
):
    
    price_grid = generate_price_grid(base_price, pct_range=pct_range, steps=steps)
    sim_df = simulate_demand(forecaster, group_key, group_id, price_grid, clip=clip, min_clip_prob=min_clip_prob)
    
    if sim_df.empty:
        print(f"No data available for subcategory {group_id}")
        return

    plt.figure(figsize=(12, 6))
    
    # Demand curve
    plt.subplot(1, 2, 1)
    plt.plot(sim_df['price'], sim_df['forecast_demand'], marker='o')
    plt.axvline(base_price, color='gray', linestyle='--', label=f'Base Price: {base_price:.2f}')
    plt.title(f"Demand Curve for {group_id}")
    plt.xlabel("Price")
    plt.ylabel("Forecasted Demand")
    plt.grid(True)
    plt.legend()
    
    # Revenue curve
    plt.subplot(1, 2, 2)
    plt.plot(sim_df['price'], sim_df['expected_revenue'], marker='o', color='green')
    plt.axvline(base_price, color='gray', linestyle='--', label=f'Base Price: {base_price:.2f}')
    plt.title(f"Revenue Curve for {group_id}")
    plt.xlabel("Price")
    plt.ylabel("Expected Revenue")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)

    if not supress_output:
        plt.show() 