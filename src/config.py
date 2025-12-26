"""
Configuration file for demand prediction model.
Contains default parameters and settings.
"""

# Model Architecture Parameters
SEQUENCE_LENGTH = 30  # Number of time steps in input sequence
LSTM_UNITS = 64  # Number of units in LSTM layers
DROPOUT_RATE = 0.2  # Dropout rate for regularization
PREDICTION_PERIODS = 1  # Number of periods ahead to predict

# Training Parameters
EPOCHS = 50  # Number of training epochs
BATCH_SIZE = 32  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for optimizer
TEST_SIZE = 0.2  # Proportion of data for testing

# Data Parameters
DATA_DIR = "data"  # Directory containing data files
FREQUENCY = "D"  # Time frequency: 'D' (daily), 'W' (weekly), 'M' (monthly)

# Model Paths
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/demand_prediction_model.h5"
PLOTS_DIR = "plots"

# Feature Columns
FEATURE_COLUMNS = [
    'order_count',
    'total_revenue',
    'total_freight',
    'day_of_week',
    'day_of_month',
    'month'
]

TARGET_COLUMN = 'order_count'  # Column to predict

# Reproducibility
RANDOM_SEED = 42

