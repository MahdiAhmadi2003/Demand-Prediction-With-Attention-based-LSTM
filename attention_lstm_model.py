"""
Attention-based LSTM Model for Demand Prediction
Implements an LSTM network with attention mechanism for time series forecasting.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class AttentionLayer(layers.Layer):
    """Attention mechanism layer for LSTM outputs."""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a = tf.nn.softmax(e, axis=1)
        output = tf.reduce_sum(a * inputs, axis=1)
        return output, a
    
    def compute_output_shape(self, input_shape):
        # Returns (output_shape, attention_weights_shape)
        output_shape = (input_shape[0], input_shape[2])
        attention_shape = (input_shape[0], input_shape[1], 1)
        return output_shape, attention_shape
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        return config


class AttentionLSTM:
    """Attention-based LSTM model for demand prediction."""
    
    def __init__(self,
                 sequence_length: int = 30,
                 n_features: int = 6,
                 lstm_units: int = 64,
                 dropout_rate: float = 0.2,
                 prediction_periods: int = 1):
        """
        Initialize the Attention-based LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            prediction_periods: Number of periods to predict ahead
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.prediction_periods = prediction_periods
        self.model = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """
        Build the attention-based LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # First LSTM layer with return_sequences=True
        lstm1 = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            activation='tanh'
        )(inputs)
        lstm1 = layers.Dropout(self.dropout_rate)(lstm1)
        
        # Second LSTM layer
        lstm2 = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            activation='tanh'
        )(lstm1)
        lstm2 = layers.Dropout(self.dropout_rate)(lstm2)
        
        # Attention layer
        attention_output, attention_weights = AttentionLayer()(lstm2)
        
        # Dense layers
        dense1 = layers.Dense(32, activation='relu')(attention_output)
        dense1 = layers.Dropout(self.dropout_rate)(dense1)
        
        dense2 = layers.Dense(16, activation='relu')(dense1)
        
        # Output layer
        if self.prediction_periods == 1:
            outputs = layers.Dense(1, activation='linear')(dense2)
        else:
            outputs = layers.Dense(self.prediction_periods, activation='linear')(dense2)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        return model
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 32,
              verbose: int = 1,
              callbacks: Optional[list] = None,
              show_progress: bool = True) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            X_val: Validation input sequences (optional)
            y_val: Validation target values (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level
            callbacks: List of Keras callbacks
            show_progress: Whether to show enhanced progress display
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
        
        # Add enhanced progress callback if requested
        if show_progress:
            try:
                from training_progress import TrainingProgressCallback, MetricsLogger
                callbacks.append(TrainingProgressCallback(total_epochs=epochs))
                callbacks.append(MetricsLogger(log_file="training_log.txt"))
            except ImportError:
                # If import fails, continue without enhanced progress
                pass
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=callbacks
        )
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate the model.
        
        Args:
            X: Input sequences
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        
        metrics = self.model.evaluate(X, y, verbose=0)
        metric_names = self.model.metrics_names
        
        return dict(zip(metric_names, metrics))
    
    def save_model(self, filepath: str):
        """Save the model to a file."""
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a model from a file."""
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"Model loaded from {filepath}")

