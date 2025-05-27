import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import json
import datetime

def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def initialize_metrics_store(storage_path: Optional[Path] = None) -> Dict:
    """
    Initialize or load an existing metrics store.
    
    Args:
        storage_path: Optional path to store metrics persistently
        
    Returns:
        Dictionary containing regression and classification metrics
    """
    metrics = {
        'regression': {},
        'classification': {}
    }
    
    if storage_path and storage_path.exists():
        with open(storage_path, 'r') as f:
            metrics = json.load(f)
            
    return metrics

def store_regression_metrics(
    metrics: Dict,
    model_name: str,
    new_metrics: Dict,
    storage_path: Optional[Path] = None
) -> Dict:
    """
    Store regression metrics for a model.
    
    Args:
        metrics: The metrics store dictionary
        model_name: Name of the model
        new_metrics: Dictionary of new metrics to store
        storage_path: Optional path to store metrics persistently
    """
    if 'regression' not in metrics:
        metrics['regression'] = {}
        
    if model_name not in metrics['regression']:
        metrics['regression'][model_name] = []
        
    # Add timestamp
    new_metrics['timestamp'] = datetime.datetime.now().isoformat()
    
    # Convert numpy types to native Python types
    new_metrics = _convert_numpy_types(new_metrics)
    
    # Store metrics
    metrics['regression'][model_name].append(new_metrics)
    
    # Save to file if path provided
    if storage_path:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(storage_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    return metrics

def store_classification_metrics(
    metrics: Dict,
    model_name: str,
    new_metrics: Dict,
    storage_path: Optional[Path] = None
) -> Dict:
    """
    Store classification metrics for a model.
    
    Args:
        metrics: The metrics store dictionary
        model_name: Name of the model
        new_metrics: Dictionary of new metrics to store
        storage_path: Optional path to store metrics persistently
    """
    if 'classification' not in metrics:
        metrics['classification'] = {}
        
    if model_name not in metrics['classification']:
        metrics['classification'][model_name] = []
        
    # Add timestamp
    new_metrics['timestamp'] = datetime.datetime.now().isoformat()
    
    # Store metrics
    metrics['classification'][model_name].append(new_metrics)
    
    # Save to file if path provided
    if storage_path:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(storage_path, 'w') as f:
            json.dump(metrics, f, indent=2)
            
    return metrics

def get_latest_metrics(
    metrics: Dict,
    model_name: str,
    metric_type: str = 'regression'
) -> Optional[Dict]:
    """
    Get the most recent metrics for a model.
    
    Args:
        metrics: The metrics store dictionary
        model_name: Name of the model
        metric_type: Type of metrics ('regression' or 'classification')
        
    Returns:
        Dictionary of latest metrics or None if not found
    """
    if metric_type not in metrics or model_name not in metrics[metric_type]:
        return None
        
    model_metrics = metrics[metric_type][model_name]
    if not model_metrics:
        return None
        
    return model_metrics[-1]

def get_metrics_history(
    metrics: Dict,
    model_name: str,
    metric_type: str = 'regression'
) -> Optional[pd.DataFrame]:
    """
    Get the full history of metrics for a model.
    
    Args:
        metrics: The metrics store dictionary
        model_name: Name of the model
        metric_type: Type of metrics ('regression' or 'classification')
        
    Returns:
        DataFrame of metrics history or None if not found
    """
    if metric_type not in metrics or model_name not in metrics[metric_type]:
        return None
        
    model_metrics = metrics[metric_type][model_name]
    if not model_metrics:
        return None
        
    return pd.DataFrame(model_metrics) 