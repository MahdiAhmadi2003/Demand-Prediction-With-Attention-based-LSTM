"""
Data Preprocessing Module for Demand Prediction
Handles loading and preprocessing of Olist e-commerce data for time series analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Preprocesses e-commerce data for demand prediction."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir: Directory containing the CSV data files
        """
        self.data_dir = data_dir
        self.demand_data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and merge order and order items data.
        
        Returns:
            Merged DataFrame with order and item information
        """
        print("Loading data files...")
        
        # Load orders dataset
        orders_df = pd.read_csv(f"{self.data_dir}/olist_orders_dataset.csv")
        orders_df['order_purchase_timestamp'] = pd.to_datetime(
            orders_df['order_purchase_timestamp']
        )
        
        # Load order items dataset
        order_items_df = pd.read_csv(f"{self.data_dir}/olist_order_items_dataset.csv")
        
        # Merge datasets
        merged_df = order_items_df.merge(
            orders_df[['order_id', 'order_purchase_timestamp', 'order_status']],
            on='order_id',
            how='inner'
        )
        
        # Filter only delivered orders
        merged_df = merged_df[merged_df['order_status'] == 'delivered']
        
        print(f"Loaded {len(merged_df)} order items from {len(orders_df)} orders")
        return merged_df
    
    def aggregate_demand(self, 
                        df: pd.DataFrame, 
                        frequency: str = 'D') -> pd.DataFrame:
        """
        Aggregate demand by time period.
        
        Args:
            df: Merged DataFrame with order data
            frequency: Time frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)
            
        Returns:
            DataFrame with aggregated demand by time period
        """
        print(f"Aggregating demand by {frequency} frequency...")
        
        # Set timestamp as index
        df = df.copy()
        df.set_index('order_purchase_timestamp', inplace=True)
        
        # Aggregate by time period
        demand_agg = df.resample(frequency).agg({
            'order_id': 'count',  # Number of orders
            'price': 'sum',  # Total revenue
            'freight_value': 'sum',  # Total freight
        }).rename(columns={
            'order_id': 'order_count',
            'price': 'total_revenue',
            'freight_value': 'total_freight'
        })
        
        # Fill missing dates with 0
        demand_agg = demand_agg.fillna(0)
        
        # Add additional features
        demand_agg['date'] = demand_agg.index
        demand_agg['day_of_week'] = demand_agg.index.dayofweek
        demand_agg['day_of_month'] = demand_agg.index.day
        demand_agg['month'] = demand_agg.index.month
        demand_agg['year'] = demand_agg.index.year
        
        print(f"Aggregated to {len(demand_agg)} time periods")
        return demand_agg
    
    def create_sequences(self, 
                        data: pd.DataFrame,
                        target_col: str = 'order_count',
                        sequence_length: int = 30,
                        prediction_periods: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Time series DataFrame
            target_col: Column name to predict
            sequence_length: Number of time steps to use as input
            prediction_periods: Number of periods ahead to predict
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        print(f"Creating sequences with length {sequence_length}...")
        
        # Select features
        feature_cols = [
            'order_count', 'total_revenue', 'total_freight',
            'day_of_week', 'day_of_month', 'month'
        ]
        
        # Normalize features
        data_normalized = data[feature_cols].copy()
        self.scalers = {}
        
        for col in feature_cols:
            if col == target_col:
                # For target, store min/max for inverse transform
                col_min = data_normalized[col].min()
                col_max = data_normalized[col].max()
                if col_max - col_min > 0:
                    data_normalized[col] = (data_normalized[col] - col_min) / (col_max - col_min)
                else:
                    data_normalized[col] = 0
                self.scalers[col] = {'min': col_min, 'max': col_max}
            else:
                # For features, use standard scaling
                col_mean = data_normalized[col].mean()
                col_std = data_normalized[col].std()
                if col_std > 0:
                    data_normalized[col] = (data_normalized[col] - col_mean) / col_std
                else:
                    data_normalized[col] = 0
                self.scalers[col] = {'mean': col_mean, 'std': col_std}
        
        # Create sequences
        X, y = [], []
        
        for i in range(len(data_normalized) - sequence_length - prediction_periods + 1):
            X.append(data_normalized.iloc[i:i+sequence_length].values)
            y.append(data_normalized[target_col].iloc[i+sequence_length:i+sequence_length+prediction_periods].values)
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape y if prediction_periods == 1
        if prediction_periods == 1:
            y = y.reshape(-1, 1)
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y
    
    def inverse_transform_target(self, y_normalized: np.ndarray) -> np.ndarray:
        """
        Inverse transform normalized target values to original scale.
        
        Args:
            y_normalized: Normalized target values
            
        Returns:
            Original scale target values
        """
        if 'order_count' in self.scalers:
            scaler = self.scalers['order_count']
            y_original = y_normalized * (scaler['max'] - scaler['min']) + scaler['min']
            return y_original
        return y_normalized
    
    def prepare_data(self,
                    frequency: str = 'D',
                    sequence_length: int = 30,
                    prediction_periods: int = 1,
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete data preparation pipeline.
        
        Args:
            frequency: Time frequency for aggregation
            sequence_length: Number of time steps for input sequences
            prediction_periods: Number of periods ahead to predict
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Load and process data
        merged_df = self.load_data()
        demand_df = self.aggregate_demand(merged_df, frequency=frequency)
        self.demand_data = demand_df
        
        # Create sequences
        X, y = self.create_sequences(
            demand_df,
            sequence_length=sequence_length,
            prediction_periods=prediction_periods
        )
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\nData split:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test

