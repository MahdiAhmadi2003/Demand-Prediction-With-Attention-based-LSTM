"""
Main entry point for the Demand Prediction System.
This is the primary script to run the demand prediction application.
"""

import sys
import os

# Add src directory to path to ensure imports work
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from demand_prediction import main

if __name__ == "__main__":
    """
    Main entry point for the application.
    
    This script serves as the primary way to run the demand prediction system.
    It delegates to the main() function in demand_prediction.py which handles
    command-line argument parsing and execution.
    
    Usage examples:
        python main.py --train
        python main.py --train --predict
        python main.py --train --predict --plot
        python main.py --predict --prediction_periods 7
    """
    main()

