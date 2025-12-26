# Data Relationships and Features Used for Training

## Overview
The demand prediction model now uses **ALL 9 CSV files** from the data folder, properly merged based on their relationships.

## Data Files and Relationships

### Core Relationship Structure:
```
orders (core table)
├── order_items (via order_id) - One order can have multiple items
│   ├── products (via product_id)
│   │   └── category_translation (via product_category_name)
│   └── sellers (via seller_id)
│       └── geolocation (via seller_zip_code_prefix)
├── customers (via customer_id)
│   └── geolocation (via customer_zip_code_prefix)
├── payments (via order_id) - One order can have multiple payments
└── reviews (via order_id) - One order can have multiple reviews
```

## Detailed File Relationships

### 1. **olist_orders_dataset.csv** (Core)
- **Primary Key**: `order_id`
- **Links to**: 
  - `order_items` (1-to-many)
  - `customers` (many-to-1 via `customer_id`)
  - `payments` (1-to-many)
  - `reviews` (1-to-many)
- **Key Fields Used**:
  - `order_purchase_timestamp` (time index)
  - `order_status` (filter: only 'delivered')

### 2. **olist_order_items_dataset.csv**
- **Primary Key**: `order_item_id`
- **Foreign Keys**: 
  - `order_id` → orders
  - `product_id` → products
  - `seller_id` → sellers
- **Key Fields Used**:
  - `price`, `freight_value`

### 3. **olist_customers_dataset.csv**
- **Primary Key**: `customer_id`
- **Links to**: 
  - `orders` (1-to-many)
  - `geolocation` (many-to-1 via `customer_zip_code_prefix`)
- **Key Fields Used**:
  - `customer_zip_code_prefix`, `customer_city`, `customer_state`

### 4. **olist_products_dataset.csv**
- **Primary Key**: `product_id`
- **Links to**: 
  - `order_items` (1-to-many)
  - `category_translation` (many-to-1 via `product_category_name`)
- **Key Fields Used**:
  - `product_category_name`, `product_weight_g`, `product_length_cm`, 
  - `product_height_cm`, `product_width_cm`, `product_photos_qty`

### 5. **olist_sellers_dataset.csv**
- **Primary Key**: `seller_id`
- **Links to**: 
  - `order_items` (1-to-many)
  - `geolocation` (many-to-1 via `seller_zip_code_prefix`)
- **Key Fields Used**:
  - `seller_zip_code_prefix`, `seller_city`, `seller_state`

### 6. **olist_order_payments_dataset.csv**
- **Primary Key**: `review_id` (implicit)
- **Foreign Key**: `order_id` → orders
- **Aggregation**: Multiple payments per order are aggregated
- **Key Fields Used**:
  - `payment_value` (summed), `payment_installments` (averaged), `payment_type` (mode)

### 7. **olist_order_reviews_dataset.csv**
- **Primary Key**: `review_id`
- **Foreign Key**: `order_id` → orders
- **Aggregation**: Multiple reviews per order are averaged
- **Key Fields Used**:
  - `review_score` (averaged)

### 8. **olist_geolocation_dataset.csv**
- **Primary Key**: `geolocation_zip_code_prefix` (with lat/lng)
- **Links to**: 
  - `customers` (via `customer_zip_code_prefix`)
  - `sellers` (via `seller_zip_code_prefix`)
- **Aggregation**: Multiple coordinates per zip code are averaged
- **Key Fields Used**:
  - `geolocation_lat`, `geolocation_lng` (averaged per zip code)

### 9. **product_category_name_translation.csv**
- **Primary Key**: `product_category_name`
- **Links to**: `products` (many-to-1)
- **Key Fields Used**:
  - `product_category_name_english` (translated category names)

## Features Created for Training

After merging all datasets, the following **27 features** are created through time-based aggregation:

### Basic Order Metrics (3 features):
- `order_count`: Number of unique orders per period
- `total_revenue`: Sum of all order prices
- `total_freight`: Sum of all freight values

### Customer Metrics (2 features):
- `unique_customers`: Number of unique customers per period
- `customer_state_code`: Encoded most common customer state

### Product Metrics (6 features):
- `unique_products`: Number of unique products per period
- `category_code`: Encoded most common product category
- `avg_product_weight`: Average product weight
- `avg_product_length`: Average product length
- `avg_product_height`: Average product height
- `avg_product_width`: Average product width

### Seller Metrics (2 features):
- `unique_sellers`: Number of unique sellers per period
- `seller_state_code`: Encoded most common seller state

### Payment Metrics (2 features):
- `total_payment_value`: Sum of all payment values
- `avg_payment_installments`: Average number of installments

### Review Metrics (1 feature):
- `avg_review_score`: Average review score

### Temporal Features (6 features):
- `day_of_week`: Day of week (0-6)
- `day_of_month`: Day of month (1-31)
- `week_of_year`: Week number (1-52)
- `month`: Month (1-12)
- `year`: Year
- `is_weekend`: Binary indicator (0 or 1)

### Derived Features (2 features):
- `avg_order_value`: Average revenue per order
- `avg_items_per_order`: Average number of items per order

## Training Configuration

- **Input Features**: 23 features (all except `order_count` which is the target)
- **Target Variable**: `order_count` (number of orders per period)
- **Sequence Length**: 30 time steps (default, configurable)
- **Prediction Periods**: 1 period ahead (default, configurable)

## Data Flow

1. **Load**: All 9 CSV files are loaded
2. **Merge**: Files are merged following the relationship structure
3. **Filter**: Only 'delivered' orders are kept
4. **Aggregate**: Data is aggregated by time period (daily/weekly/monthly)
5. **Encode**: Categorical variables are encoded as numeric codes
6. **Normalize**: Features are standardized, target is min-max normalized
7. **Sequence**: Time series sequences are created for LSTM training

## Benefits of Using All Datasets

1. **Richer Features**: More information about customers, products, sellers, and transactions
2. **Better Patterns**: Model can learn from product categories, geographic patterns, payment methods, and review scores
3. **Improved Accuracy**: More features typically lead to better demand predictions
4. **Comprehensive Analysis**: All aspects of the e-commerce business are considered

