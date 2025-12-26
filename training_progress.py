"""
Custom callback for enhanced training progress display.
"""

import tensorflow as tf
from tensorflow import keras
import time
import os


class TrainingProgressCallback(keras.callbacks.Callback):
    """Enhanced callback for displaying training progress."""
    
    def __init__(self, total_epochs: int):
        """
        Initialize the progress callback.
        
        Args:
            total_epochs: Total number of training epochs
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = []
        
    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self.start_time = time.time()
        print("\n" + "=" * 80)
        print("TRAINING STARTED")
        print("=" * 80)
        print(f"Total epochs: {self.total_epochs}")
        print(f"Training samples: {self.params.get('samples', 'N/A')}")
        print(f"Batch size: {self.params.get('batch_size', 'N/A')}")
        print("=" * 80 + "\n")
        
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch."""
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{self.total_epochs}")
        print("-" * 80)
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - (epoch + 1)
        estimated_remaining = avg_time * remaining_epochs
        
        print("-" * 80)
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        
        if logs:
            print("\nMetrics:")
            for key, value in logs.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
        
        print(f"\nProgress: {epoch + 1}/{self.total_epochs} epochs ({100 * (epoch + 1) / self.total_epochs:.1f}%)")
        print(f"Average time per epoch: {avg_time:.2f}s")
        print(f"Estimated time remaining: {estimated_remaining / 60:.1f} minutes")
        print("=" * 80)
        
    def on_train_end(self, logs=None):
        """Called at the end of training."""
        total_time = time.time() - self.start_time
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Total training time: {total_time / 60:.2f} minutes ({total_time:.2f} seconds)")
        print(f"Average time per epoch: {sum(self.epoch_times) / len(self.epoch_times):.2f} seconds")
        print(f"Total epochs: {len(self.epoch_times)}")
        
        if logs:
            print("\nFinal Metrics:")
            for key, value in logs.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.6f}")
        
        print("=" * 80 + "\n")


class MetricsLogger(keras.callbacks.Callback):
    """Callback to log metrics to a file."""
    
    def __init__(self, log_file: str = "training_log.txt"):
        """
        Initialize the metrics logger.
        
        Args:
            log_file: Path to log file
        """
        super().__init__()
        self.log_file = log_file
        self.metrics_log = []
        
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        log_entry = {
            'epoch': epoch + 1,
            **{k: float(v) for k, v in logs.items() if isinstance(v, (int, float))}
        }
        self.metrics_log.append(log_entry)
        
        # Write to file
        with open(self.log_file, 'w') as f:
            f.write("Epoch, Loss, MAE, MAPE")
            if 'val_loss' in logs:
                f.write(", Val_Loss, Val_MAE, Val_MAPE")
            f.write("\n")
            
            for entry in self.metrics_log:
                f.write(f"{entry['epoch']}, {entry.get('loss', 'N/A')}, "
                       f"{entry.get('mae', 'N/A')}, {entry.get('mape', 'N/A')}")
                if 'val_loss' in entry:
                    f.write(f", {entry.get('val_loss', 'N/A')}, "
                           f"{entry.get('val_mae', 'N/A')}, {entry.get('val_mape', 'N/A')}")
                f.write("\n")

