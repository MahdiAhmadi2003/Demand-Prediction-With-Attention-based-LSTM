"""
Location-based Demand Prediction Module
Handles training and prediction for demand by city/state with detailed Excel reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import DataPreprocessor
from attention_lstm_model import AttentionLSTM
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


class LocationBasedPredictor:
    """Handles location-based demand prediction by city or state."""
    
    def __init__(self, 
                 data_dir: str = "data",
                 location_type: str = "state",
                 min_samples_per_location: int = 100):
        """
        Initialize location-based predictor.
        
        Args:
            data_dir: Directory containing data files
            location_type: 'state' or 'city' for grouping
            min_samples_per_location: Minimum samples required to train a location-specific model
        """
        self.data_dir = data_dir
        self.location_type = location_type.lower()
        self.min_samples_per_location = min_samples_per_location
        self.preprocessor = DataPreprocessor(data_dir=data_dir)
        self.models: Dict[str, AttentionLSTM] = {}
        self.location_data: Dict[str, pd.DataFrame] = {}
        self.location_stats: Dict[str, dict] = {}
        
        if self.location_type not in ['state', 'city']:
            raise ValueError("location_type must be 'state' or 'city'")
    
    def prepare_location_data(self, frequency: str = 'D') -> Dict[str, pd.DataFrame]:
        """
        Prepare data grouped by location (city or state).
        
        Args:
            frequency: Time frequency for aggregation
            
        Returns:
            Dictionary mapping location names to aggregated DataFrames
        """
        print(f"\n{'='*60}")
        print(f"Preparing Location-Based Data (by {self.location_type.upper()})")
        print(f"{'='*60}")
        
        # Load and merge all data
        merged_df = self.preprocessor.load_data()
        
        # Determine location column
        if self.location_type == 'state':
            location_col = 'customer_state'
        else:  # city
            location_col = 'customer_city'
        
        # Filter out missing locations
        merged_df = merged_df[merged_df[location_col].notna()]
        
        # Get unique locations
        locations = merged_df[location_col].unique()
        print(f"\nFound {len(locations)} unique {self.location_type}s")
        
        location_data = {}
        location_stats = {}
        
        for location in locations:
            print(f"\nProcessing {self.location_type}: {location}")
            
            # Filter data for this location
            location_df = merged_df[merged_df[location_col] == location].copy()
            
            # Aggregate demand for this location
            try:
                demand_df = self.preprocessor.aggregate_demand(location_df, frequency=frequency)
                # Store demand_data for frequency inference
                self.preprocessor.demand_data = demand_df
                
                # Check if we have enough data
                if len(demand_df) < self.min_samples_per_location:
                    print(f"  Warning: Only {len(demand_df)} samples for {location}. Skipping.")
                    continue
                
                # Store location data
                location_data[location] = demand_df
                
                # Calculate statistics
                stats = {
                    'total_samples': len(demand_df),
                    'date_range': (demand_df.index.min(), demand_df.index.max()),
                    'avg_daily_orders': demand_df['order_count'].mean(),
                    'total_orders': demand_df['order_count'].sum(),
                    'total_revenue': demand_df['total_revenue'].sum(),
                    'has_sufficient_data': True
                }
                location_stats[location] = stats
                
                print(f"  Samples: {stats['total_samples']}, Avg Daily Orders: {stats['avg_daily_orders']:.2f}")
                
            except Exception as e:
                print(f"  Error processing {location}: {str(e)}")
                location_stats[location] = {
                    'has_sufficient_data': False,
                    'error': str(e)
                }
                continue
        
        self.location_data = location_data
        self.location_stats = location_stats
        
        print(f"\n{'='*60}")
        print(f"Successfully prepared data for {len(location_data)} {self.location_type}s")
        print(f"{'='*60}")
        
        return location_data
    
    def train_location_models(self,
                             sequence_length: int = 30,
                             prediction_periods: int = 1,
                             epochs: int = 50,
                             batch_size: int = 32,
                             test_size: float = 0.2,
                             lstm_units: int = 128) -> Dict[str, AttentionLSTM]:
        """
        Train separate models for each location.
        
        Args:
            sequence_length: Length of input sequences
            prediction_periods: Number of periods to predict ahead
            epochs: Number of training epochs
            batch_size: Batch size for training
            test_size: Proportion of data for testing
            lstm_units: Number of LSTM units
            
        Returns:
            Dictionary mapping location names to trained models
        """
        print(f"\n{'='*60}")
        print(f"Training Location-Based Models")
        print(f"{'='*60}")
        
        if not self.location_data:
            raise ValueError("No location data prepared. Call prepare_location_data() first.")
        
        models = {}
        training_results = {}
        
        for location, demand_df in self.location_data.items():
            print(f"\n{'='*40}")
            print(f"Training model for {self.location_type}: {location}")
            print(f"{'='*40}")
            
            try:
                # Create sequences
                X, y = self.preprocessor.create_sequences(
                    demand_df,
                    sequence_length=sequence_length,
                    prediction_periods=prediction_periods
                )
                
                if len(X) < 50:  # Need minimum sequences
                    print(f"  Insufficient sequences ({len(X)}). Skipping.")
                    training_results[location] = {
                        'status': 'insufficient_data',
                        'sequences': len(X)
                    }
                    continue
                
                # Split data
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Get number of features
                n_features = X_train.shape[2]
                
                # Initialize model
                model = AttentionLSTM(
                    sequence_length=sequence_length,
                    n_features=n_features,
                    lstm_units=lstm_units,
                    dropout_rate=0.3,
                    prediction_periods=prediction_periods,
                    use_batch_norm=True,
                    use_residual=True
                )
                
                # Build model
                model.build_model()
                
                # Train model
                history = model.train(
                    X_train, y_train,
                    X_val=X_test, y_val=y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0,  # Less verbose for multiple models
                    show_progress=False
                )
                
                # Evaluate
                test_metrics = model.evaluate(X_test, y_test)
                
                # Make predictions for evaluation
                y_pred = model.predict(X_test)
                y_pred_original = self.preprocessor.inverse_transform_target(y_pred)
                y_test_original = self.preprocessor.inverse_transform_target(y_test)
                
                # Calculate additional metrics
                mae = np.mean(np.abs(y_pred_original - y_test_original))
                rmse = np.sqrt(np.mean((y_pred_original - y_test_original) ** 2))
                mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-8))) * 100
                
                models[location] = model
                
                training_results[location] = {
                    'status': 'success',
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'final_loss': history.history['loss'][-1] if history else None,
                    'final_val_loss': history.history['val_loss'][-1] if history else None
                }
                
                print(f"  Training completed!")
                print(f"  Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
                
            except Exception as e:
                print(f"  Error training model for {location}: {str(e)}")
                training_results[location] = {
                    'status': 'error',
                    'error': str(e)
                }
                continue
        
        self.models = models
        self.training_results = training_results
        
        print(f"\n{'='*60}")
        print(f"Successfully trained {len(models)} location models")
        print(f"{'='*60}")
        
        return models
    
    def predict_all_locations(self, 
                            sequence_length: int = 30,
                            prediction_periods: int = 1) -> Dict[str, dict]:
        """
        Make predictions for all locations.
        
        Args:
            sequence_length: Length of input sequences
            prediction_periods: Number of periods to predict ahead
            
        Returns:
            Dictionary with predictions for each location
        """
        print(f"\n{'='*60}")
        print(f"Making Predictions for All Locations")
        print(f"{'='*60}")
        
        predictions = {}
        
        for location, model in self.models.items():
            print(f"\nPredicting for {self.location_type}: {location}")
            
            try:
                demand_df = self.location_data[location]
                
                # Create sequences
                X, _ = self.preprocessor.create_sequences(
                    demand_df,
                    sequence_length=sequence_length,
                    prediction_periods=prediction_periods
                )
                
                if len(X) == 0:
                    print(f"  No sequences available. Skipping.")
                    continue
                
                # Use last sequence for prediction
                last_sequence = X[-1:].reshape(1, sequence_length, X.shape[2])
                
                # Make prediction
                prediction_normalized = model.predict(last_sequence)
                prediction = self.preprocessor.inverse_transform_target(prediction_normalized)
                
                # Get last date
                last_date = demand_df.index[-1]
                
                # Calculate prediction dates
                if self.preprocessor.demand_data is not None:
                    freq = pd.infer_freq(self.preprocessor.demand_data.index)
                else:
                    freq = 'D'
                
                if freq == 'D' or freq is None:
                    next_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                              periods=prediction_periods, freq='D')
                elif freq == 'W':
                    next_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), 
                                              periods=prediction_periods, freq='W')
                else:  # Monthly
                    next_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                              periods=prediction_periods, freq='M')
                
                predictions[location] = {
                    'predictions': prediction[0],
                    'dates': next_dates,
                    'last_actual_date': last_date,
                    'last_actual_value': demand_df['order_count'].iloc[-1],
                    'avg_historical': demand_df['order_count'].mean(),
                    'status': 'success'
                }
                
                print(f"  Predicted demand: {prediction[0][0]:.2f} orders")
                
            except Exception as e:
                print(f"  Error predicting for {location}: {str(e)}")
                predictions[location] = {
                    'status': 'error',
                    'error': str(e)
                }
                continue
        
        return predictions
    
    def generate_excel_report(self, 
                             predictions: Dict[str, dict],
                             output_file: str = "reports/location_demand_report.xlsx"):
        """
        Generate detailed Excel report with predictions and statistics.
        
        Args:
            predictions: Dictionary with predictions for each location
            output_file: Output Excel file path
        """
        print(f"\n{'='*60}")
        print(f"Generating Excel Report")
        print(f"{'='*60}")
        
        # Ensure reports directory exists
        report_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else 'reports'
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
            print(f"Created reports directory: {report_dir}")
        
        wb = Workbook()
        
        # Remove default sheet
        if 'Sheet' in wb.sheetnames:
            wb.remove(wb['Sheet'])
        
        # Define styles
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=11)
        title_font = Font(bold=True, size=14)
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Sheet 1: Summary
        ws_summary = wb.create_sheet("Summary", 0)
        self._create_summary_sheet(ws_summary, predictions, header_fill, header_font, 
                                  title_font, border)
        
        # Sheet 2: Predictions by Location
        ws_predictions = wb.create_sheet("Predictions by Location", 1)
        self._create_predictions_sheet(ws_predictions, predictions, header_fill, 
                                      header_font, title_font, border)
        
        # Sheet 3: Model Performance
        ws_performance = wb.create_sheet("Model Performance", 2)
        self._create_performance_sheet(ws_performance, header_fill, header_font, 
                                      title_font, border)
        
        # Sheet 4: Location Statistics
        ws_stats = wb.create_sheet("Location Statistics", 3)
        self._create_statistics_sheet(ws_stats, header_fill, header_font, 
                                     title_font, border)
        
        # Save workbook
        wb.save(output_file)
        print(f"\nExcel report saved to: {output_file}")
        
        return output_file
    
    def _create_summary_sheet(self, ws, predictions, header_fill, header_font, 
                             title_font, border):
        """Create summary sheet."""
        ws['A1'] = f"Demand Prediction Report - By {self.location_type.upper()}"
        ws['A1'].font = title_font
        ws.merge_cells('A1:D1')
        
        ws['A3'] = "Generated on:"
        ws['B3'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Headers
        headers = ['Location', 'Predicted Demand', 'Last Actual', 'Change %', 'Status']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=5, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border
        
        # Data
        row = 6
        for location, pred_data in sorted(predictions.items()):
            if pred_data.get('status') == 'success':
                predicted = pred_data['predictions'][0] if len(pred_data['predictions']) > 0 else 0
                last_actual = pred_data.get('last_actual_value', 0)
                change_pct = ((predicted - last_actual) / (last_actual + 1e-8)) * 100 if last_actual > 0 else 0
                
                ws.cell(row=row, column=1, value=location).border = border
                ws.cell(row=row, column=2, value=f"{predicted:.2f}").border = border
                ws.cell(row=row, column=3, value=f"{last_actual:.2f}").border = border
                ws.cell(row=row, column=4, value=f"{change_pct:.2f}%").border = border
                ws.cell(row=row, column=5, value="Success").border = border
                row += 1
        
        # Auto-adjust column widths
        for col in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 20
    
    def _create_predictions_sheet(self, ws, predictions, header_fill, header_font, 
                                 title_font, border):
        """Create detailed predictions sheet."""
        ws['A1'] = f"Detailed Predictions by {self.location_type.upper()}"
        ws['A1'].font = title_font
        ws.merge_cells('A1:E1')
        
        headers = ['Location', 'Prediction Date', 'Predicted Orders', 
                  'Historical Average', 'Confidence Level']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border
        
        row = 4
        for location, pred_data in sorted(predictions.items()):
            if pred_data.get('status') == 'success':
                dates = pred_data['dates']
                preds = pred_data['predictions']
                avg_hist = pred_data.get('avg_historical', 0)
                
                for i, (date, pred) in enumerate(zip(dates, preds)):
                    ws.cell(row=row, column=1, value=location).border = border
                    ws.cell(row=row, column=2, value=date.strftime("%Y-%m-%d")).border = border
                    ws.cell(row=row, column=3, value=f"{pred:.2f}").border = border
                    ws.cell(row=row, column=4, value=f"{avg_hist:.2f}").border = border
                    # Simple confidence based on model performance
                    confidence = "High" if location in self.models else "Medium"
                    ws.cell(row=row, column=5, value=confidence).border = border
                    row += 1
        
        for col in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 25
    
    def _create_performance_sheet(self, ws, header_fill, header_font, title_font, border):
        """Create model performance sheet."""
        ws['A1'] = "Model Performance Metrics"
        ws['A1'].font = title_font
        ws.merge_cells('A1:F1')
        
        headers = ['Location', 'Train Samples', 'Test Samples', 'MAE', 'RMSE', 'MAPE %']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border
        
        row = 4
        if hasattr(self, 'training_results'):
            for location, results in sorted(self.training_results.items()):
                if results.get('status') == 'success':
                    ws.cell(row=row, column=1, value=location).border = border
                    ws.cell(row=row, column=2, value=results.get('train_samples', 0)).border = border
                    ws.cell(row=row, column=3, value=results.get('test_samples', 0)).border = border
                    ws.cell(row=row, column=4, value=f"{results.get('mae', 0):.2f}").border = border
                    ws.cell(row=row, column=5, value=f"{results.get('rmse', 0):.2f}").border = border
                    ws.cell(row=row, column=6, value=f"{results.get('mape', 0):.2f}").border = border
                    row += 1
        
        for col in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 18
    
    def _create_statistics_sheet(self, ws, header_fill, header_font, title_font, border):
        """Create location statistics sheet."""
        ws['A1'] = f"Location Statistics - {self.location_type.upper()}"
        ws['A1'].font = title_font
        ws.merge_cells('A1:G1')
        
        headers = ['Location', 'Total Samples', 'Date Range Start', 'Date Range End',
                  'Avg Daily Orders', 'Total Orders', 'Total Revenue']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.border = border
        
        row = 4
        for location, stats in sorted(self.location_stats.items()):
            if stats.get('has_sufficient_data'):
                ws.cell(row=row, column=1, value=location).border = border
                ws.cell(row=row, column=2, value=stats.get('total_samples', 0)).border = border
                date_range = stats.get('date_range', (None, None))
                ws.cell(row=row, column=3, value=str(date_range[0]) if date_range[0] else '').border = border
                ws.cell(row=row, column=4, value=str(date_range[1]) if date_range[1] else '').border = border
                ws.cell(row=row, column=5, value=f"{stats.get('avg_daily_orders', 0):.2f}").border = border
                ws.cell(row=row, column=6, value=f"{stats.get('total_orders', 0):.2f}").border = border
                ws.cell(row=row, column=7, value=f"{stats.get('total_revenue', 0):.2f}").border = border
                row += 1
        
        for col in range(1, len(headers) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 20

