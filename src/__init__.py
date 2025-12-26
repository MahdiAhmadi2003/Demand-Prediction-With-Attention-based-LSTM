"""
Source code package for Demand Prediction with Attention-based LSTM.

This package contains all the core modules for demand prediction:
- attention_lstm_model: Enhanced LSTM model with attention mechanism
- data_preprocessing: Data loading and preprocessing utilities
- demand_prediction: Main prediction and training functions
- location_based_prediction: Location-based prediction module
- training_progress: Training progress display utilities
- config: Configuration parameters
"""

__version__ = "1.0.0"

# Import main components for easier access
# Note: Using absolute imports since files are in same directory
try:
    from attention_lstm_model import AttentionLSTM, EnsemblePredictor
    from data_preprocessing import DataPreprocessor
    from demand_prediction import train_model, predict_demand, main
    from location_based_prediction import LocationBasedPredictor
except ImportError:
    # Fallback for when used as package
    from .attention_lstm_model import AttentionLSTM, EnsemblePredictor
    from .data_preprocessing import DataPreprocessor
    from .demand_prediction import train_model, predict_demand, main
    from .location_based_prediction import LocationBasedPredictor

__all__ = [
    'AttentionLSTM',
    'EnsemblePredictor',
    'DataPreprocessor',
    'LocationBasedPredictor',
    'train_model',
    'predict_demand',
    'main'
]

