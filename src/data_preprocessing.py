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
        Load and merge all datasets with proper relationships.
        
        Relationship structure:
        - orders (core) → order_items, customers, payments, reviews
        - order_items → products, sellers
        - products → category_translation
        - customers → geolocation (via zip_code)
        - sellers → geolocation (via zip_code)
        
        Returns:
            Merged DataFrame with all relevant information
        """
        print("Loading all data files...")
        
        # 1. Load core datasets
        print("  Loading orders dataset...")
        orders_df = pd.read_csv(f"{self.data_dir}/olist_orders_dataset.csv")
        orders_df['order_purchase_timestamp'] = pd.to_datetime(
            orders_df['order_purchase_timestamp']
        )
        
        print("  Loading order items dataset...")
        order_items_df = pd.read_csv(f"{self.data_dir}/olist_order_items_dataset.csv")
        
        # 2. Merge orders with order_items (one-to-many: one order can have multiple items)
        print("  Merging orders with order items...")
        merged_df = order_items_df.merge(
            orders_df[['order_id', 'customer_id', 'order_purchase_timestamp', 'order_status']],
            on='order_id',
            how='inner'
        )
        
        # Filter only delivered orders
        merged_df = merged_df[merged_df['order_status'] == 'delivered']
        print(f"  After filtering delivered orders: {len(merged_df)} order items")
        
        # 3. Merge with customers (via customer_id from orders)
        print("  Loading and merging customers dataset...")
        customers_df = pd.read_csv(f"{self.data_dir}/olist_customers_dataset.csv")
        merged_df = merged_df.merge(
            customers_df[['customer_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state']],
            on='customer_id',
            how='left'
        )
        
        # 4. Merge with products (via product_id from order_items)
        print("  Loading and merging products dataset...")
        products_df = pd.read_csv(f"{self.data_dir}/olist_products_dataset.csv")
        
        # Merge with category translation
        print("  Loading and merging product category translation...")
        category_translation_df = pd.read_csv(f"{self.data_dir}/product_category_name_translation.csv")
        products_df = products_df.merge(
            category_translation_df,
            on='product_category_name',
            how='left'
        )
        
        # Merge products with order_items
        merged_df = merged_df.merge(
            products_df[[
                'product_id', 'product_category_name_english', 
                'product_weight_g', 'product_length_cm', 'product_height_cm', 
                'product_width_cm', 'product_photos_qty'
            ]],
            on='product_id',
            how='left'
        )
        
        # 5. Merge with sellers (via seller_id from order_items)
        print("  Loading and merging sellers dataset...")
        sellers_df = pd.read_csv(f"{self.data_dir}/olist_sellers_dataset.csv")
        merged_df = merged_df.merge(
            sellers_df[['seller_id', 'seller_zip_code_prefix', 'seller_city', 'seller_state']],
            on='seller_id',
            how='left'
        )
        
        # 6. Merge with payments (via order_id from orders)
        print("  Loading and merging payments dataset...")
        payments_df = pd.read_csv(f"{self.data_dir}/olist_order_payments_dataset.csv")
        # Aggregate payments per order (one order can have multiple payments)
        payments_agg = payments_df.groupby('order_id').agg({
            'payment_value': 'sum',
            'payment_installments': 'mean',
            'payment_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
        }).reset_index()
        payments_agg.columns = ['order_id', 'total_payment_value', 'avg_payment_installments', 'payment_type']
        
        merged_df = merged_df.merge(
            payments_agg,
            on='order_id',
            how='left'
        )
        
        # 7. Merge with reviews (via order_id from orders)
        print("  Loading and merging reviews dataset...")
        reviews_df = pd.read_csv(f"{self.data_dir}/olist_order_reviews_dataset.csv")
        # Aggregate reviews per order (one order can have multiple reviews)
        reviews_agg = reviews_df.groupby('order_id').agg({
            'review_score': 'mean',
            'review_creation_date': 'first'
        }).reset_index()
        reviews_agg.columns = ['order_id', 'avg_review_score', 'review_creation_date']
        
        merged_df = merged_df.merge(
            reviews_agg,
            on='order_id',
            how='left'
        )
        
        # 8. Merge with geolocation for customers (via customer_zip_code_prefix)
        print("  Loading and merging customer geolocation...")
        geolocation_df = pd.read_csv(f"{self.data_dir}/olist_geolocation_dataset.csv")
        # Get average lat/lng per zip code (one zip can have multiple coordinates)
        geo_customer = geolocation_df.groupby('geolocation_zip_code_prefix').agg({
            'geolocation_lat': 'mean',
            'geolocation_lng': 'mean'
        }).reset_index()
        geo_customer.columns = ['customer_zip_code_prefix', 'customer_lat', 'customer_lng']
        
        merged_df = merged_df.merge(
            geo_customer,
            on='customer_zip_code_prefix',
            how='left'
        )
        
        # 9. Merge with geolocation for sellers (via seller_zip_code_prefix)
        print("  Loading and merging seller geolocation...")
        geo_seller = geolocation_df.groupby('geolocation_zip_code_prefix').agg({
            'geolocation_lat': 'mean',
            'geolocation_lng': 'mean'
        }).reset_index()
        geo_seller.columns = ['seller_zip_code_prefix', 'seller_lat', 'seller_lng']
        
        merged_df = merged_df.merge(
            geo_seller,
            on='seller_zip_code_prefix',
            how='left'
        )
        
        print(f"\nSuccessfully loaded and merged all datasets!")
        print(f"Final dataset shape: {merged_df.shape}")
        print(f"Columns: {len(merged_df.columns)}")
        
        return merged_df
    
    def aggregate_demand(self, 
                        df: pd.DataFrame, 
                        frequency: str = 'D') -> pd.DataFrame:
        """
        Aggregate demand by time period with features from all datasets.
        
        Args:
            df: Merged DataFrame with all data
            frequency: Time frequency for aggregation ('D'=daily, 'W'=weekly, 'M'=monthly)
            
        Returns:
            DataFrame with aggregated demand and features by time period
        """
        print(f"Aggregating demand by {frequency} frequency with all features...")
        
        # Set timestamp as index
        df = df.copy()
        df.set_index('order_purchase_timestamp', inplace=True)
        
        # Define aggregation functions for different columns
        agg_dict = {
            # Basic order metrics
            'order_id': 'nunique',  # Number of unique orders
            'price': 'sum',  # Total revenue
            'freight_value': 'sum',  # Total freight
            
            # Customer metrics
            'customer_id': 'nunique',  # Number of unique customers
            'customer_state': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',  # Most common state
            
            # Product metrics
            'product_id': 'nunique',  # Number of unique products
            'product_category_name_english': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',  # Most common category
            'product_weight_g': 'mean',  # Average product weight
            'product_length_cm': 'mean',  # Average product length
            'product_height_cm': 'mean',  # Average product height
            'product_width_cm': 'mean',  # Average product width
            
            # Seller metrics
            'seller_id': 'nunique',  # Number of unique sellers
            'seller_state': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown',  # Most common seller state
            
            # Payment metrics
            'total_payment_value': 'sum',  # Total payment value
            'avg_payment_installments': 'mean',  # Average installments
            
            # Review metrics
            'avg_review_score': 'mean',  # Average review score
        }
        
        # Aggregate by time period
        demand_agg = df.resample(frequency).agg(agg_dict)
        
        # Rename columns for clarity
        demand_agg.columns = [
            'order_count', 'total_revenue', 'total_freight',
            'unique_customers', 'most_common_customer_state',
            'unique_products', 'most_common_category',
            'avg_product_weight', 'avg_product_length', 'avg_product_height', 'avg_product_width',
            'unique_sellers', 'most_common_seller_state',
            'total_payment_value', 'avg_payment_installments',
            'avg_review_score'
        ]
        
        # Fill missing dates with 0 for numeric columns
        numeric_cols = demand_agg.select_dtypes(include=[np.number]).columns
        demand_agg[numeric_cols] = demand_agg[numeric_cols].fillna(0)
        
        # Fill missing categorical columns
        categorical_cols = ['most_common_customer_state', 'most_common_category', 'most_common_seller_state']
        for col in categorical_cols:
            if col in demand_agg.columns:
                demand_agg[col] = demand_agg[col].fillna('unknown')
        
        # Encode categorical variables as numeric (for time series)
        # Convert states to numeric codes and handle missing values
        if 'most_common_customer_state' in demand_agg.columns:
            cat = pd.Categorical(demand_agg['most_common_customer_state'])
            demand_agg['customer_state_code'] = cat.codes
            # Replace -1 (missing) with 0
            demand_agg['customer_state_code'] = demand_agg['customer_state_code'].replace(-1, 0)
        if 'most_common_seller_state' in demand_agg.columns:
            cat = pd.Categorical(demand_agg['most_common_seller_state'])
            demand_agg['seller_state_code'] = cat.codes
            # Replace -1 (missing) with 0
            demand_agg['seller_state_code'] = demand_agg['seller_state_code'].replace(-1, 0)
        if 'most_common_category' in demand_agg.columns:
            cat = pd.Categorical(demand_agg['most_common_category'])
            demand_agg['category_code'] = cat.codes
            # Replace -1 (missing) with 0
            demand_agg['category_code'] = demand_agg['category_code'].replace(-1, 0)
        
        # Drop original categorical columns (keep only numeric codes)
        demand_agg = demand_agg.drop(columns=categorical_cols, errors='ignore')
        
        # Add temporal features
        demand_agg['day_of_week'] = demand_agg.index.dayofweek.astype(np.float64)
        demand_agg['day_of_month'] = demand_agg.index.day.astype(np.float64)
        # Handle week_of_year properly
        try:
            demand_agg['week_of_year'] = demand_agg.index.isocalendar().week.astype(np.float64)
        except:
            # Fallback if isocalendar() doesn't work
            demand_agg['week_of_year'] = demand_agg.index.strftime('%U').astype(np.float64)
        demand_agg['month'] = demand_agg.index.month.astype(np.float64)
        demand_agg['year'] = demand_agg.index.year.astype(np.float64)
        demand_agg['is_weekend'] = (demand_agg.index.dayofweek >= 5).astype(np.float64)
        
        # Calculate additional derived features
        demand_agg['avg_order_value'] = np.where(
            demand_agg['order_count'] > 0,
            demand_agg['total_revenue'] / demand_agg['order_count'],
            0
        ).astype(np.float64)
        demand_agg['avg_items_per_order'] = np.where(
            demand_agg['order_count'] > 0,
            demand_agg['unique_products'] / demand_agg['order_count'],
            0
        ).astype(np.float64)
        
        # Feature Engineering: Lag features (previous periods)
        print("  Creating lag features...")
        lag_periods = [1, 2, 3, 7, 14]  # 1 day, 2 days, 3 days, 1 week, 2 weeks
        for lag in lag_periods:
            demand_agg[f'order_count_lag_{lag}'] = demand_agg['order_count'].shift(lag).fillna(0).astype(np.float64)
            demand_agg[f'total_revenue_lag_{lag}'] = demand_agg['total_revenue'].shift(lag).fillna(0).astype(np.float64)
        
        # Feature Engineering: Rolling statistics (moving averages and std)
        print("  Creating rolling statistics...")
        rolling_windows = [3, 7, 14, 30]  # 3 days, 1 week, 2 weeks, 1 month
        for window in rolling_windows:
            # Rolling mean
            demand_agg[f'order_count_ma_{window}'] = demand_agg['order_count'].rolling(
                window=window, min_periods=1
            ).mean().fillna(0).astype(np.float64)
            demand_agg[f'total_revenue_ma_{window}'] = demand_agg['total_revenue'].rolling(
                window=window, min_periods=1
            ).mean().fillna(0).astype(np.float64)
            
            # Rolling standard deviation
            demand_agg[f'order_count_std_{window}'] = demand_agg['order_count'].rolling(
                window=window, min_periods=1
            ).std().fillna(0).astype(np.float64)
            
            # Rolling max and min
            demand_agg[f'order_count_max_{window}'] = demand_agg['order_count'].rolling(
                window=window, min_periods=1
            ).max().fillna(0).astype(np.float64)
            demand_agg[f'order_count_min_{window}'] = demand_agg['order_count'].rolling(
                window=window, min_periods=1
            ).min().fillna(0).astype(np.float64)
        
        # Feature Engineering: Trend features (difference from moving average)
        demand_agg['order_count_trend_7'] = (
            demand_agg['order_count'] - demand_agg['order_count_ma_7']
        ).astype(np.float64)
        demand_agg['order_count_trend_30'] = (
            demand_agg['order_count'] - demand_agg['order_count_ma_30']
        ).astype(np.float64)
        
        # Feature Engineering: Rate of change
        demand_agg['order_count_roc'] = (
            demand_agg['order_count'].pct_change(periods=1).fillna(0) * 100
        ).astype(np.float64)
        
        # Feature Engineering: Seasonal patterns (day of week, month interactions)
        demand_agg['weekend_order_ratio'] = np.where(
            demand_agg['is_weekend'] == 1,
            demand_agg['order_count'] / (demand_agg['order_count_ma_7'] + 1),
            0
        ).astype(np.float64)
        
        # Ensure all columns are numeric (convert any remaining object types)
        for col in demand_agg.columns:
            if demand_agg[col].dtype == 'object':
                # Try to convert to numeric, if fails, drop the column
                try:
                    demand_agg[col] = pd.to_numeric(demand_agg[col], errors='coerce').fillna(0)
                except:
                    print(f"Warning: Dropping non-numeric column: {col}")
                    demand_agg = demand_agg.drop(columns=[col])
        
        # Final check: ensure all columns are numeric
        non_numeric = demand_agg.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            print(f"Warning: Dropping non-numeric columns: {list(non_numeric)}")
            demand_agg = demand_agg.drop(columns=non_numeric)
        
        # Convert all to float64 to ensure compatibility with Keras
        demand_agg = demand_agg.astype(np.float64)
        
        print(f"Aggregated to {len(demand_agg)} time periods")
        print(f"Features created: {len(demand_agg.columns)}")
        print(f"All columns are numeric: {all(demand_agg.dtypes.apply(lambda x: np.issubdtype(x, np.number)))}")
        
        return demand_agg
    
    def create_sequences(self, 
                        data: pd.DataFrame,
                        target_col: str = 'order_count',
                        sequence_length: int = 30,
                        prediction_periods: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training with all available features.
        
        Args:
            data: Time series DataFrame
            target_col: Column name to predict
            sequence_length: Number of time steps to use as input
            prediction_periods: Number of periods ahead to predict
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        print(f"Creating sequences with length {sequence_length}...")
        
        # Select all numeric features (excluding target for input, but including it for normalization)
        # We'll exclude target from X but keep it for proper scaling
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Define feature columns (all numeric features except target will be used as input)
        # Target will be used for output only
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        # Add target to feature_cols temporarily for normalization, then remove it from X
        feature_cols_with_target = feature_cols + [target_col]
        
        # Check which columns actually exist in the data
        available_cols = [col for col in feature_cols_with_target if col in data.columns]
        
        if len(available_cols) == 0:
            raise ValueError("No valid feature columns found in the data!")
        
        if target_col not in available_cols:
            raise ValueError(f"Target column '{target_col}' not found in data!")
        
        # Remove target from input features
        input_feature_cols = [col for col in available_cols if col != target_col]
        
        print(f"Using {len(input_feature_cols)} input features:")
        print(f"  Features: {', '.join(input_feature_cols[:10])}{'...' if len(input_feature_cols) > 10 else ''}")
        
        # Normalize features (including target for proper scaling)
        data_normalized = data[available_cols].copy()
        self.scalers = {}
        
        for col in available_cols:
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
        
        # Get indices for input features (excluding target)
        input_feature_indices = [available_cols.index(col) for col in input_feature_cols]
        target_index = available_cols.index(target_col)
        
        for i in range(len(data_normalized) - sequence_length - prediction_periods + 1):
            # X: input features only (exclude target)
            X.append(data_normalized.iloc[i:i+sequence_length, input_feature_indices].values)
            # y: target values
            y.append(data_normalized[target_col].iloc[i+sequence_length:i+sequence_length+prediction_periods].values)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # Reshape y if prediction_periods == 1
        if prediction_periods == 1:
            y = y.reshape(-1, 1)
        
        # Final check: ensure no NaN or inf values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("Warning: NaN or Inf values found in X, replacing with 0")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            print("Warning: NaN or Inf values found in y, replacing with 0")
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
        print(f"Number of features per time step: {X.shape[2]}")
        
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

