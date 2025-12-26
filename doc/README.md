# Documentation Index

This folder contains detailed documentation for the Demand Prediction with Attention-based LSTM project.

## Available Documentation

### 1. [DATA_RELATIONSHIPS.md](DATA_RELATIONSHIPS.md)
**Complete guide to data relationships and features**

- Overview of all 9 CSV files used
- Detailed relationship structure between datasets
- Complete list of 27+ features created
- Data flow and preprocessing pipeline
- Benefits of using all datasets

**Read this if you want to understand:**
- How data files are related
- What features are used for training
- How data is merged and processed

### 2. [MODEL_ENHANCEMENTS.md](MODEL_ENHANCEMENTS.md)
**Detailed explanation of model architecture enhancements**

- Enhanced attention mechanisms (multi-head attention)
- Batch normalization and residual connections
- Deeper architecture improvements
- Feature engineering enhancements (58 features)
- Training improvements (Huber loss, better scheduling)
- Ensemble capabilities

**Read this if you want to understand:**
- Model architecture details
- Why the model performs better
- Feature engineering techniques
- Training optimizations

### 3. [LOCATION_BASED_GUIDE.md](LOCATION_BASED_GUIDE.md)
**Complete guide for location-based prediction**

- How to use location-based prediction
- Command-line arguments
- Excel report structure
- Troubleshooting tips
- Best practices

**Read this if you want to:**
- Predict demand by city or state
- Generate Excel reports
- Understand location-based features

## Quick Reference

### Standard Prediction
```bash
python main.py --train --predict --plot
```

### Location-Based Prediction (State)
```bash
python main.py --train --location_based --location_type state
```

### Location-Based Prediction (City)
```bash
python main.py --train --location_based --location_type city
```

## Documentation Structure

```
doc/
├── README.md                    # This file - documentation index
├── DATA_RELATIONSHIPS.md        # Data relationships and features
├── MODEL_ENHANCEMENTS.md        # Model architecture and enhancements
└── LOCATION_BASED_GUIDE.md      # Location-based prediction guide
```

## Getting Started

1. **New to the project?** Start with the main [README.md](../README.md)
2. **Want to understand the data?** Read [DATA_RELATIONSHIPS.md](DATA_RELATIONSHIPS.md)
3. **Interested in model details?** Read [MODEL_ENHANCEMENTS.md](MODEL_ENHANCEMENTS.md)
4. **Need location-based predictions?** Read [LOCATION_BASED_GUIDE.md](LOCATION_BASED_GUIDE.md)

## Questions?

If you have questions about:
- **Data**: Check [DATA_RELATIONSHIPS.md](DATA_RELATIONSHIPS.md)
- **Model**: Check [MODEL_ENHANCEMENTS.md](MODEL_ENHANCEMENTS.md)
- **Location Features**: Check [LOCATION_BASED_GUIDE.md](LOCATION_BASED_GUIDE.md)
- **General Usage**: Check the main [README.md](../README.md)

