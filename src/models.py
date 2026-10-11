"""Deep-learning architectures used for movement-behaviour classification."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GRU,
    Input,
    LSTM,
    MaxPooling1D,
)


def build_lstm(input_shape=(1000, 4), num_classes=4, dropout=0.2):
    """Four-layer LSTM model."""
    return Sequential([
        Input(shape=input_shape),
        LSTM(256, return_sequences=True),
        Dropout(dropout),
        BatchNormalization(),
        LSTM(128, return_sequences=True),
        Dropout(dropout),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(dropout),
        BatchNormalization(),
        LSTM(32),
        Dropout(dropout),
        BatchNormalization(),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="lstm")


def build_bilstm(input_shape=(1000, 4), num_classes=4, dropout=0.2):
    """Four-layer bidirectional LSTM model."""
    return Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(dropout),
        BatchNormalization(),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(dropout),
        BatchNormalization(),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(dropout),
        BatchNormalization(),
        Bidirectional(LSTM(32)),
        Dropout(dropout),
        BatchNormalization(),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="bilstm")


def build_gru(input_shape=(1000, 4), num_classes=4, dropout=0.2):
    """Four-layer GRU model."""
    return Sequential([
        Input(shape=input_shape),
        GRU(256, return_sequences=True),
        Dropout(dropout),
        BatchNormalization(),
        GRU(128, return_sequences=True),
        Dropout(dropout),
        BatchNormalization(),
        GRU(64, return_sequences=True),
        Dropout(dropout),
        BatchNormalization(),
        GRU(32),
        Dropout(dropout),
        BatchNormalization(),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="gru")


def build_cnn(input_shape=(1000, 4), num_classes=4, dropout=0.2):
    """Four-layer 1D-CNN model."""
    return Sequential([
        Input(shape=input_shape),
        Conv1D(256, 3, activation="relu"),
        MaxPooling1D(2),
        Dropout(dropout),
        BatchNormalization(),
        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),
        Dropout(dropout),
        BatchNormalization(),
        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),
        Dropout(dropout),
        BatchNormalization(),
        Conv1D(32, 3, activation="relu"),
        MaxPooling1D(2),
        Dropout(dropout),
        BatchNormalization(),
        Flatten(),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ], name="cnn")
