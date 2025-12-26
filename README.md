# Demand Prediction using Attention-based LSTM

A deep learning solution for predicting demand in e-commerce using an Attention-based Long Short-Term Memory (LSTM) neural network. This project uses real Olist e-commerce dataset to forecast future demand periods.

## Features

- **Attention-based LSTM Architecture**: Implements a sophisticated neural network with attention mechanism for better time series forecasting
- **Flexible Prediction Periods**: Predict demand for the next period(s) (configurable parameter)
- **Multiple Time Frequencies**: Support for daily, weekly, and monthly aggregation
- **Comprehensive Evaluation**: Includes MAE, RMSE, MAPE metrics and visualization
- **Production Ready**: Well-structured code following software engineering best practices

## Project Structure

```
.
├── data/                          # Data directory (CSV files)
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   └── ...
├── main.py                        # Main entry point (run this to start)
├── demand_prediction.py           # Core script for training and prediction
├── data_preprocessing.py          # Data loading and preprocessing module
├── attention_lstm_model.py        # Attention-based LSTM model implementation
├── training_progress.py           # Enhanced training progress display
├── config.py                      # Configuration parameters
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd code
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start (Recommended)

Use `main.py` as the entry point:

```bash
# Train and predict with plots
python main.py --train --predict --plot

# Train only
python main.py --train

# Predict only (requires trained model)
python main.py --predict
```

### Training a Model

Train a new model with default parameters:
```bash
python main.py --train
```

Train with custom parameters:
```bash
python main.py --train \
    --prediction_periods 7 \
    --sequence_length 30 \
    --frequency D \
    --epochs 100 \
    --batch_size 32 \
    --lstm_units 128
```

### Making Predictions

After training, make predictions:
```bash
python main.py --predict
```

Or train and predict in one go:
```bash
python main.py --train --predict
```

### Generate Plots

Include visualization plots:
```bash
python main.py --train --plot
```

### Training Progress

The training process displays:
- **Real-time progress**: Epoch-by-epoch progress with metrics
- **Time estimates**: Estimated time remaining for training
- **Metrics tracking**: Loss, MAE, MAPE for both training and validation
- **Training log**: Automatic logging to `training_log.txt`

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prediction_periods` | int | 1 | Number of periods ahead to predict |
| `--sequence_length` | int | 30 | Length of input sequences |
| `--frequency` | str | D | Time frequency: D (daily), W (weekly), M (monthly) |
| `--epochs` | int | 50 | Number of training epochs |
| `--batch_size` | int | 32 | Batch size for training |
| `--lstm_units` | int | 64 | Number of LSTM units |
| `--test_size` | float | 0.2 | Proportion of data for testing |
| `--model_path` | str | models/demand_prediction_model.h5 | Path to save/load model |
| `--train` | flag | - | Train a new model |
| `--predict` | flag | - | Make predictions |
| `--plot` | flag | - | Generate prediction plots |

## Model Architecture

The model consists of:
1. **Input Layer**: Accepts sequences of historical demand data
2. **LSTM Layers**: Two stacked LSTM layers with dropout for regularization
3. **Attention Layer**: Attention mechanism to focus on important time steps
4. **Dense Layers**: Fully connected layers for final prediction
5. **Output Layer**: Linear activation for regression task

## Data Preprocessing

The preprocessing pipeline:
1. Loads and merges order and order items data
2. Filters only delivered orders
3. Aggregates demand by time period (daily/weekly/monthly)
4. Creates sequences for LSTM input
5. Normalizes features and target variables
6. Splits data into training and testing sets

## Output Files

- `models/demand_prediction_model.h5`: Trained model file
- `models/demand_prediction_model_history.json`: Training history
- `predictions.csv`: Predicted demand values
- `plots/training_history.png`: Training loss and metrics plots
- `plots/predictions_vs_actual.png`: Comparison of predictions vs actual values
- `plots/scatter_plot.png`: Scatter plot of predictions vs actual

## Evaluation Metrics

The model is evaluated using:
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences
- **MAPE (Mean Absolute Percentage Error)**: Percentage error metric

## Example Workflow

1. **Prepare your data**: Ensure CSV files are in the `data/` directory
2. **Train the model**:
   ```bash
   python main.py --train --plot
   ```
3. **Monitor training progress**: Watch the detailed progress display showing:
   - Current epoch and total epochs
   - Training and validation metrics
   - Time elapsed and estimated time remaining
   - Progress percentage
4. **Review the results**: Check the plots in `plots/` directory
5. **Make predictions**:
   ```bash
   python main.py --predict
   ```
6. **Check predictions**: View `predictions.csv` for forecasted demand
7. **Review training log**: Check `training_log.txt` for detailed metrics history

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy
- Pandas
- Matplotlib

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Author

Created for demand forecasting in e-commerce applications.

## Notes

- The model uses early stopping and learning rate reduction for better training
- Model checkpoints are saved automatically during training
- Predictions are saved in CSV format for easy integration with other systems
- The attention mechanism helps the model focus on relevant time steps in the sequence

