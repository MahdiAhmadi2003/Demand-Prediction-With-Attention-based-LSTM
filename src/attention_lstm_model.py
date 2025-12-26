"""
Enhanced Attention-based LSTM Model for Demand Prediction
Implements an advanced LSTM network with multi-head attention, batch normalization,
residual connections, and ensemble capabilities for improved time series forecasting.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, List


class MultiHeadAttentionLayer(layers.Layer):
    """Multi-head attention mechanism for better feature extraction."""
    
    def __init__(self, num_heads: int = 4, **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
    
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.head_dim = feature_dim // self.num_heads
        
        # Query, Key, Value projections for each head
        self.W_q = self.add_weight(
            name='query_weight',
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_k = self.add_weight(
            name='key_weight',
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_v = self.add_weight(
            name='value_weight',
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.W_o = self.add_weight(
            name='output_weight',
            shape=(feature_dim, feature_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        super(MultiHeadAttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        feature_dim = inputs.shape[-1]
        
        # Project to Q, K, V
        Q = tf.matmul(inputs, self.W_q)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)
        
        # Reshape for multi-head attention
        Q = tf.reshape(Q, [batch_size, seq_len, self.num_heads, self.head_dim])
        K = tf.reshape(K, [batch_size, seq_len, self.num_heads, self.head_dim])
        V = tf.reshape(V, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose for attention computation
        Q = tf.transpose(Q, [0, 2, 1, 3])  # [batch, heads, seq, dim]
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        attended = tf.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = tf.transpose(attended, [0, 2, 1, 3])  # [batch, seq, heads, dim]
        attended = tf.reshape(attended, [batch_size, seq_len, feature_dim])
        
        # Output projection
        output = tf.matmul(attended, self.W_o)
        
        return output, attention_weights


class AttentionLayer(layers.Layer):
    """Enhanced attention mechanism layer with additive attention."""
    
    def __init__(self, use_additive: bool = True, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.use_additive = use_additive
    
    def build(self, input_shape):
        feature_dim = input_shape[-1]
        if self.use_additive:
            # Additive attention (Bahdanau style)
            self.W1 = self.add_weight(
                name='attention_W1',
                shape=(feature_dim, feature_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            self.W2 = self.add_weight(
                name='attention_W2',
                shape=(feature_dim, feature_dim),
                initializer='glorot_uniform',
                trainable=True
            )
            self.v = self.add_weight(
                name='attention_v',
                shape=(feature_dim, 1),
                initializer='glorot_uniform',
                trainable=True
            )
        else:
            # Dot-product attention
            self.W = self.add_weight(
                name='attention_weight',
                shape=(feature_dim, 1),
                initializer='glorot_uniform',
                trainable=True
            )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, features)
        if self.use_additive:
            # Additive attention
            # Compute attention scores
            hidden = tf.tanh(tf.matmul(inputs, self.W1))
            scores = tf.matmul(hidden, self.v)  # [batch, time_steps, 1]
            attention_weights = tf.nn.softmax(scores, axis=1)
            # Weighted sum
            output = tf.reduce_sum(attention_weights * inputs, axis=1)
        else:
            # Dot-product attention
            e = tf.matmul(inputs, self.W)  # [batch, time_steps, 1]
            attention_weights = tf.nn.softmax(e, axis=1)
            output = tf.reduce_sum(attention_weights * inputs, axis=1)
        
        return output, attention_weights
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'use_additive': self.use_additive})
        return config


class AttentionLSTM:
    """Enhanced Attention-based LSTM model for demand prediction."""
    
    def __init__(self,
                 sequence_length: int = 30,
                 n_features: int = 6,
                 lstm_units: int = 128,
                 dropout_rate: float = 0.3,
                 prediction_periods: int = 1,
                 use_batch_norm: bool = True,
                 use_residual: bool = True,
                 use_multihead_attention: bool = False,
                 num_attention_heads: int = 4):
        """
        Initialize the Enhanced Attention-based LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            n_features: Number of input features
            lstm_units: Number of LSTM units (increased default)
            dropout_rate: Dropout rate for regularization
            prediction_periods: Number of periods to predict ahead
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
            use_multihead_attention: Whether to use multi-head attention
            num_attention_heads: Number of attention heads if using multi-head
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.prediction_periods = prediction_periods
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        self.use_multihead_attention = use_multihead_attention
        self.num_attention_heads = num_attention_heads
        self.model = None
        self.history = None
    
    def build_model(self) -> keras.Model:
        """
        Build the enhanced attention-based LSTM model architecture.
        
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        x = inputs
        
        # Input normalization
        if self.use_batch_norm:
            x = layers.BatchNormalization()(x)
        
        # First LSTM layer with return_sequences=True
        lstm1 = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal'
        )(x)
        if self.use_batch_norm:
            lstm1 = layers.BatchNormalization()(lstm1)
        lstm1 = layers.Dropout(self.dropout_rate)(lstm1)
        
        # Second LSTM layer with residual connection
        lstm2 = layers.LSTM(
            self.lstm_units,
            return_sequences=True,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal'
        )(lstm1)
        if self.use_batch_norm:
            lstm2 = layers.BatchNormalization()(lstm2)
        
        # Residual connection (if dimensions match)
        if self.use_residual and lstm1.shape[-1] == lstm2.shape[-1]:
            lstm2 = layers.Add()([lstm1, lstm2])
        
        lstm2 = layers.Dropout(self.dropout_rate)(lstm2)
        
        # Third LSTM layer for deeper representation
        lstm3 = layers.LSTM(
            self.lstm_units // 2,  # Reduced units for deeper layer
            return_sequences=True,
            activation='tanh',
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal'
        )(lstm2)
        if self.use_batch_norm:
            lstm3 = layers.BatchNormalization()(lstm3)
        lstm3 = layers.Dropout(self.dropout_rate)(lstm3)
        
        # Attention mechanism
        if self.use_multihead_attention:
            attention_output, attention_weights = MultiHeadAttentionLayer(
                num_heads=self.num_attention_heads
            )(lstm3)
        else:
            attention_output, attention_weights = AttentionLayer(use_additive=True)(lstm3)
        
        # Batch normalization after attention
        if self.use_batch_norm:
            attention_output = layers.BatchNormalization()(attention_output)
        
        # Dense layers with residual connections
        dense1 = layers.Dense(128, activation='relu')(attention_output)
        if self.use_batch_norm:
            dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(self.dropout_rate)(dense1)
        
        dense2 = layers.Dense(64, activation='relu')(dense1)
        if self.use_batch_norm:
            dense2 = layers.BatchNormalization()(dense2)
        dense2 = layers.Dropout(self.dropout_rate * 0.5)(dense2)
        
        dense3 = layers.Dense(32, activation='relu')(dense2)
        if self.use_batch_norm:
            dense3 = layers.BatchNormalization()(dense3)
        dense3 = layers.Dropout(self.dropout_rate * 0.5)(dense3)
        
        # Output layer
        if self.prediction_periods == 1:
            outputs = layers.Dense(1, activation='linear', name='output')(dense3)
        else:
            outputs = layers.Dense(self.prediction_periods, activation='linear', name='output')(dense3)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model with improved optimizer and loss
        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            ),
            loss=self._huber_loss,  # Use Huber loss for robustness
            metrics=['mae', self._smape_metric]  # Add SMAPE metric
        )
        
        self.model = model
        return model
    
    def _huber_loss(self, y_true, y_pred, delta=1.0):
        """Huber loss function - more robust to outliers than MSE."""
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= delta
        squared_loss = tf.square(error) / 2
        linear_loss = delta * tf.abs(error) - tf.square(delta) / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    
    def _smape_metric(self, y_true, y_pred):
        """Symmetric Mean Absolute Percentage Error metric."""
        numerator = tf.abs(y_true - y_pred)
        denominator = (tf.abs(y_true) + tf.abs(y_pred)) / 2.0
        return tf.reduce_mean(numerator / (denominator + 1e-8)) * 100
    
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
        
        # Default callbacks with enhanced learning rate scheduling
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=1e-6
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.3,
                    patience=7,
                    min_lr=1e-8,
                    verbose=1,
                    mode='min'
                ),
                # Learning rate scheduler
                keras.callbacks.LearningRateScheduler(
                    lambda epoch: 0.001 * (0.9 ** epoch),
                    verbose=0
                )
            ]
            # Remove None callbacks
            callbacks = [cb for cb in callbacks if cb is not None]
        
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
        custom_objects = {
            'AttentionLayer': AttentionLayer,
            'MultiHeadAttentionLayer': MultiHeadAttentionLayer,
            '_huber_loss': self._huber_loss,
            '_smape_metric': self._smape_metric
        }
        self.model = keras.models.load_model(
            filepath,
            custom_objects=custom_objects
        )
        print(f"Model loaded from {filepath}")


class EnsemblePredictor:
    """Ensemble of multiple models for improved predictions."""
    
    def __init__(self, models: List[AttentionLSTM]):
        """
        Initialize ensemble with multiple models.
        
        Args:
            models: List of trained AttentionLSTM models
        """
        self.models = models
    
    def predict(self, X: np.ndarray, method: str = 'mean') -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input sequences
            method: Ensemble method ('mean', 'median', 'weighted')
            
        Returns:
            Ensemble predictions
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # [n_models, n_samples, n_outputs]
        
        if method == 'mean':
            return np.mean(predictions, axis=0)
        elif method == 'median':
            return np.median(predictions, axis=0)
        elif method == 'weighted':
            # Weight by inverse of variance (more weight to confident models)
            weights = 1.0 / (np.var(predictions, axis=0) + 1e-8)
            weights = weights / np.sum(weights, axis=0, keepdims=True)
            return np.sum(predictions * weights, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, method: str = 'mean') -> dict:
        """
        Evaluate ensemble on test data.
        
        Args:
            X: Input sequences
            y: True target values
            method: Ensemble method
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X, method=method)
        
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

