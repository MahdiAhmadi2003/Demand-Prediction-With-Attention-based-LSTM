# Location-Based Demand Prediction Guide

## Overview
This feature enables demand prediction by **city** or **state**, training separate models for each location and generating a comprehensive Excel report.

## Features

### 1. **Location-Based Training**
- Automatically groups data by city or state
- Trains separate models for each location with sufficient data
- Filters out locations with insufficient samples (configurable minimum)

### 2. **Comprehensive Excel Report**
The generated Excel report contains 4 detailed sheets:

#### Sheet 1: Summary
- Quick overview of all locations
- Predicted demand vs last actual value
- Percentage change
- Status of each prediction

#### Sheet 2: Predictions by Location
- Detailed predictions for each location
- Prediction dates
- Historical averages for comparison
- Confidence levels

#### Sheet 3: Model Performance
- Training and test sample counts
- Performance metrics (MAE, RMSE, MAPE) for each location model
- Model quality indicators

#### Sheet 4: Location Statistics
- Total samples per location
- Date ranges covered
- Average daily orders
- Total orders and revenue

## Usage

### Basic Usage (By State)
```bash
python main.py --train --location_based --location_type state
```

### By City
```bash
python main.py --train --location_based --location_type city
```

### With Custom Parameters
```bash
python main.py --train --location_based --location_type state \
    --epochs 30 \
    --batch_size 32 \
    --min_location_samples 150 \
    --report_file my_report.xlsx
```

## Command Line Arguments

### Location-Based Arguments
- `--location_based`: Enable location-based prediction
- `--location_type`: Choose 'state' or 'city' (default: 'state')
- `--min_location_samples`: Minimum samples required per location (default: 100)
- `--report_file`: Output Excel file name (default: 'location_demand_report.xlsx')

### Standard Arguments (also work with location-based)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 32)
- `--sequence_length`: Input sequence length (default: 30)
- `--prediction_periods`: Number of periods to predict (default: 1)
- `--frequency`: Time frequency - D (daily), W (weekly), M (monthly) (default: D)
- `--lstm_units`: Number of LSTM units (default: 64, minimum 128 for location-based)

## How It Works

### 1. Data Preparation
- Loads all datasets and merges them
- Groups data by selected location type (city or state)
- Filters locations with insufficient data
- Aggregates demand for each location separately

### 2. Model Training
- Trains a separate Attention-based LSTM model for each location
- Uses the same enhanced architecture (batch norm, residuals, etc.)
- Evaluates each model on test data
- Stores performance metrics

### 3. Prediction
- Makes predictions for all trained locations
- Uses the last sequence from each location's data
- Calculates prediction dates based on frequency

### 4. Report Generation
- Creates comprehensive Excel workbook
- Includes formatted tables with headers and borders
- Provides multiple views of the data
- Easy to analyze and share

## Output Files

### Excel Report Structure

**Summary Sheet:**
- Location name
- Predicted demand
- Last actual value
- Percentage change
- Status

**Predictions Sheet:**
- Location
- Prediction date
- Predicted orders
- Historical average
- Confidence level

**Performance Sheet:**
- Location
- Training samples
- Test samples
- MAE, RMSE, MAPE metrics

**Statistics Sheet:**
- Location
- Total samples
- Date range
- Average daily orders
- Total orders and revenue

## Example Output

After running location-based prediction, you'll see:
```
============================================================
Location-Based Demand Prediction (STATE)
============================================================

Found 27 unique states

Processing state: SP
  Samples: 450, Avg Daily Orders: 125.32

Processing state: RJ
  Samples: 320, Avg Daily Orders: 89.45

...

Successfully trained 15 location models

Excel Report: location_demand_report.xlsx
```

## Tips

1. **State vs City**: 
   - States typically have more data per location
   - Cities provide more granular predictions
   - Start with states if you have limited data

2. **Minimum Samples**:
   - Increase `--min_location_samples` for more reliable models
   - Decrease if you want predictions for more locations
   - Recommended: 100-200 samples minimum

3. **Training Time**:
   - Location-based training takes longer (multiple models)
   - Reduce `--epochs` for faster training during testing
   - Use fewer locations initially to test

4. **Excel Report**:
   - Open in Excel or compatible spreadsheet software
   - All sheets are formatted and ready for analysis
   - Can be easily shared with stakeholders

## Troubleshooting

### "Insufficient data" warnings
- Some locations may not have enough data
- Increase `--min_location_samples` threshold
- Or use state-level instead of city-level

### Long training time
- Reduce number of epochs: `--epochs 20`
- Reduce batch size: `--batch_size 16`
- Filter to specific locations (future enhancement)

### Missing locations in report
- Check minimum samples threshold
- Verify data has location information
- Check console output for warnings

## Future Enhancements

- Filter to specific locations
- Compare predictions across locations
- Visualizations in Excel
- Export to other formats (CSV, JSON)
- Ensemble predictions across locations

