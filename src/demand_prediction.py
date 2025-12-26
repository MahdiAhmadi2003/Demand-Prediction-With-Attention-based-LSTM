"""
Main script for demand prediction using Attention-based LSTM
Predicts demand for the next period(s) based on historical data.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import tensorflow as tf

from data_preprocessing import DataPreprocessor
from attention_lstm_model import AttentionLSTM
from location_based_prediction import LocationBasedPredictor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Demand Prediction using Attention-based LSTM'
    )
    parser.add_argument(
        '--prediction_periods',
        type=int,
        default=1,
        help='Number of periods ahead to predict (default: 1)'
    )
    parser.add_argument(
        '--sequence_length',
        type=int,
        default=30,
        help='Length of input sequences (default: 30)'
    )
    parser.add_argument(
        '--frequency',
        type=str,
        default='D',
        choices=['D', 'W', 'M'],
        help='Time frequency for aggregation: D=daily, W=weekly, M=monthly (default: D)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )
    parser.add_argument(
        '--lstm_units',
        type=int,
        default=64,
        help='Number of LSTM units (default: 64)'
    )
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/demand_prediction_model.h5',
        help='Path to save/load model (default: models/demand_prediction_model.h5)'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new model'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Make predictions using trained model'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate prediction plots'
    )
    parser.add_argument(
        '--location_based',
        action='store_true',
        help='Enable location-based prediction (by city or state)'
    )
    parser.add_argument(
        '--location_type',
        type=str,
        default='state',
        choices=['state', 'city'],
        help='Location type for grouping: state or city (default: state)'
    )
    parser.add_argument(
        '--min_location_samples',
        type=int,
        default=100,
        help='Minimum samples required per location to train a model (default: 100)'
    )
    parser.add_argument(
        '--report_file',
        type=str,
        default='reports/location_demand_report.xlsx',
        help='Output Excel report file name (default: reports/location_demand_report.xlsx)'
    )
    
    return parser.parse_args()


def train_model(args):
    """Train the attention-based LSTM model."""
    print("=" * 60)
    print("Training Attention-based LSTM Model for Demand Prediction")
    print("=" * 60)
    
    # Initialize data preprocessor
    preprocessor = DataPreprocessor(data_dir="data")
    
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(
        frequency=args.frequency,
        sequence_length=args.sequence_length,
        prediction_periods=args.prediction_periods,
        test_size=args.test_size
    )
    
    # Get number of features
    n_features = X_train.shape[2]
    
    # Initialize enhanced model with better defaults
    model = AttentionLSTM(
        sequence_length=args.sequence_length,
        n_features=n_features,
        lstm_units=args.lstm_units if args.lstm_units > 64 else 128,  # Use at least 128 units
        dropout_rate=0.3,  # Slightly higher dropout for regularization
        prediction_periods=args.prediction_periods,
        use_batch_norm=True,  # Enable batch normalization
        use_residual=True,  # Enable residual connections
        use_multihead_attention=False,  # Can enable for even better performance
        num_attention_heads=4
    )
    
    # Build model
    print("\nBuilding model architecture...")
    model.build_model()
    model.model.summary()
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Add model checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        args.model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    print("\nTraining model...")
    history = model.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=[checkpoint_callback],
        show_progress=True
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_metrics = model.evaluate(X_test, y_test)
    print("\nTest Set Metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions
    y_pred_original = preprocessor.inverse_transform_target(y_pred)
    y_test_original = preprocessor.inverse_transform_target(y_test)
    
    # Calculate additional metrics
    mae = np.mean(np.abs(y_pred_original - y_test_original))
    rmse = np.sqrt(np.mean((y_pred_original - y_test_original) ** 2))
    mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-8))) * 100
    
    print(f"\nAdditional Metrics:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Save training history
    history_path = args.model_path.replace('.h5', '_history.json')
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    print(f"\nTraining history saved to {history_path}")
    
    # Plot results if requested
    if args.plot:
        plot_results(history, y_test_original, y_pred_original, args)
    
    return model, preprocessor


def predict_demand(args, model=None, preprocessor=None):
    """Make demand predictions for the next period(s)."""
    print("=" * 60)
    print("Making Demand Predictions")
    print("=" * 60)
    
    # Load and prepare data if preprocessor not provided (needed to determine n_features)
    if preprocessor is None:
        preprocessor = DataPreprocessor(data_dir="data")
        merged_df = preprocessor.load_data()
        demand_df = preprocessor.aggregate_demand(merged_df, frequency=args.frequency)
        preprocessor.demand_data = demand_df
    
    # Determine number of features from the data
    # Create a temporary sequence to get the feature count
    X_temp, _ = preprocessor.create_sequences(
        preprocessor.demand_data,
        sequence_length=args.sequence_length,
        prediction_periods=args.prediction_periods
    )
    n_features = X_temp.shape[2]
    
    # Load model if not provided
    if model is None:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(
                f"Model not found at {args.model_path}. Please train a model first using --train flag."
            )
        
        # Rebuild the model structure with correct number of features
        model = AttentionLSTM(
            sequence_length=args.sequence_length,
            n_features=n_features,  # Dynamic number of features from data
            lstm_units=args.lstm_units,
            dropout_rate=0.2,
            prediction_periods=args.prediction_periods
        )
        model.load_model(args.model_path)
    
    # Get sequences for prediction (we already have X_temp from above, but recreate to ensure consistency)
    X, _ = preprocessor.create_sequences(
        preprocessor.demand_data,
        sequence_length=args.sequence_length,
        prediction_periods=args.prediction_periods
    )
    
    if len(X) == 0:
        raise ValueError("Not enough data to create sequences for prediction.")
    
    # Use the last sequence
    last_sequence = X[-1:].reshape(1, args.sequence_length, X.shape[2])
    
    # Make prediction
    prediction_normalized = model.predict(last_sequence)
    
    # Inverse transform
    prediction = preprocessor.inverse_transform_target(prediction_normalized)
    
    print(f"\nPredicted demand for next {args.prediction_periods} period(s):")
    for i, pred in enumerate(prediction[0]):
        period_label = f"Period {i+1}" if args.prediction_periods > 1 else "Next period"
        print(f"  {period_label}: {pred:.2f} orders")
    
    # Get the last date from the data
    last_date = preprocessor.demand_data.index[-1]
    
    # Calculate next period dates
    if args.frequency == 'D':
        next_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=args.prediction_periods, freq='D')
    elif args.frequency == 'W':
        next_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=args.prediction_periods, freq='W')
    else:  # 'M'
        next_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=args.prediction_periods, freq='M')
    
    # Create prediction DataFrame
    prediction_df = pd.DataFrame({
        'date': next_dates,
        'predicted_demand': prediction[0]
    })
    
    # Ensure reports directory exists
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)
    
    # Save predictions
    predictions_path = os.path.join(reports_dir, 'predictions.csv')
    prediction_df.to_csv(predictions_path, index=False)
    print(f"\nPredictions saved to {predictions_path}")
    
    return prediction, prediction_df


def plot_results(history, y_test, y_pred, args):
    """Plot training history and prediction results."""
    print("\nGenerating plots...")
    
    # Ensure reports directory exists for plots
    os.makedirs('reports', exist_ok=True)
    
    # Plot 1: Training history
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title('Model MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('reports/training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved to reports/training_history.png")
    
    # Plot 2: Predictions vs Actual
    plt.figure(figsize=(12, 6))
    
    # Flatten if multi-step prediction
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
    else:
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()
    
    # Plot last 100 points for better visualization
    n_points = min(100, len(y_test_flat))
    indices = range(len(y_test_flat) - n_points, len(y_test_flat))
    
    plt.plot(indices, y_test_flat[indices], label='Actual', marker='o', markersize=3)
    plt.plot(indices, y_pred_flat[indices], label='Predicted', marker='s', markersize=3)
    plt.title('Predictions vs Actual (Last 100 Points)')
    plt.xlabel('Sample Index')
    plt.ylabel('Demand (Number of Orders)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    print("Predictions vs actual plot saved to reports/predictions_vs_actual.png")
    
    # Plot 3: Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_flat, y_pred_flat, alpha=0.5)
    plt.plot([y_test_flat.min(), y_test_flat.max()], 
             [y_test_flat.min(), y_test_flat.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.title('Predicted vs Actual Demand')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/scatter_plot.png', dpi=300, bbox_inches='tight')
    print("Scatter plot saved to reports/scatter_plot.png")
    
    plt.close('all')


def train_location_models(args):
    """Train location-based models and generate Excel report."""
    print("=" * 60)
    print(f"Location-Based Demand Prediction ({args.location_type.upper()})")
    print("=" * 60)
    
    # Initialize location-based predictor
    predictor = LocationBasedPredictor(
        data_dir="data",
        location_type=args.location_type,
        min_samples_per_location=args.min_location_samples
    )
    
    # Prepare location data
    predictor.prepare_location_data(frequency=args.frequency)
    
    # Train models for each location
    predictor.train_location_models(
        sequence_length=args.sequence_length,
        prediction_periods=args.prediction_periods,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        lstm_units=args.lstm_units if args.lstm_units > 64 else 128
    )
    
    # Make predictions for all locations
    predictions = predictor.predict_all_locations(
        sequence_length=args.sequence_length,
        prediction_periods=args.prediction_periods
    )
    
    # Ensure reports directory exists
    reports_dir = os.path.dirname(args.report_file) if os.path.dirname(args.report_file) else 'reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir, exist_ok=True)
    
    # Generate Excel report
    report_file = predictor.generate_excel_report(
        predictions=predictions,
        output_file=args.report_file
    )
    
    print(f"\n{'='*60}")
    print(f"Location-Based Prediction Complete!")
    print(f"Excel Report: {report_file}")
    print(f"{'='*60}")
    
    return predictor, predictions


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Handle location-based prediction
    if args.location_based:
        if args.train:
            train_location_models(args)
        else:
            print("Location-based prediction requires --train flag")
            print("Use: python main.py --train --location_based --location_type state")
    else:
        # Standard prediction
        if args.train:
            model, preprocessor = train_model(args)
            
            # If predict flag is also set, make predictions
            if args.predict:
                predict_demand(args, model=model, preprocessor=preprocessor)
        elif args.predict:
            predict_demand(args)
        else:
            print("Please specify --train or --predict (or both)")
            print("Use --help for more information")
            print("\nFor location-based prediction, use:")
            print("  python main.py --train --location_based --location_type state")


if __name__ == "__main__":
    main()

