import numpy as np
import pandas as pd

from colorama import Fore, Style
from typing import Tuple

from tensorflow import keras
from keras import Model, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, Dropout
from src.ml_logic.registry import load_model

from src.params import *


# Initialize model
def initialize_model(input_shape: tuple):
    """
    Initialize an improved Neural Network model.

    Args:
        input_shape (tuple): The shape of the input data.

    Returns:
        Model: Initialized neural network model.
    """

    model = Sequential()

    # Input layer
    model.add(Dense(256, activation="relu", input_shape=input_shape))
    model.add(BatchNormalization(momentum=0.9))

    # Hidden layers
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))  # Dropout regularization

    # Add more hidden layers as needed
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(1, activation="linear"))

    print("✅ Model initialized")

    return model



# Compile model
def compile_model(model: Model, learning_rate: int) -> Model:
    """
    Compile the Neural Network.

    Args:
        model (Model): The neural network model to be compiled.
        learning_rate (float): The learning rate for optimization.

    Returns:
        Model: Compiled neural network model.
    """

    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])


    print("✅ Model compiled")

    return model


# Train the model
def train_model(model: Model, 
                X: np.ndarray, 
                y: np.ndarray, 
                batch_size: int,
                validation_data: Tuple[np.ndarray, np.ndarray] = None) -> Tuple[Model, dict]:
    
    """
    Fit the model and return a tuple (fitted_model, history).

    Args:
        model (Model): The neural network model to be trained.
        X (np.ndarray): The input features for training.
        y (np.ndarray): The target values for training.
        batch_size (int): Batch size for training.
        validation_data (Tuple[np.ndarray, np.ndarray]): Validation data as a tuple (X_val, y_val).

    Returns:
        Tuple[Model, dict]: A tuple containing the trained model and training history.
    """

    callbacks = [EarlyStopping(monitor="val_loss", 
                               patience=5, 
                               restore_best_weights=True, 
                               verbose=0),

                ReduceLROnPlateau(factor=0.2, 
                                  patience=2, 
                                  monitor='val_loss')]

    history = model.fit(X,
                        y,
                        validation_data=validation_data,
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=0)

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")


    return model, history



# Evaluate model
def evaluate_model(model: Model, 
                   X: np.ndarray, 
                   y: np.ndarray) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on dataset

    Args:
        model (Model): The trained machine learning model to evaluate.
        X (np.ndarray): Feature dataset.
        y (np.ndarray): Target dataset.

    Returns:
        Tuple[Model, dict]: A tuple containing the trained model and a dictionary of evaluation metrics.
    """


    # Print a message to indicate that the evaluation is starting.
    print(Fore.BLUE + f"\nEvaluate model on {len(X)} rows..." + Style.RESET_ALL)

    # Check if the provided model is not None.
    if model is None:
        print(f"\n❌ no model to evaluate")
        return None

    # Call the evaluate method on the model, passing the feature dataset and target dataset.
    metrics = model.evaluate(x=X, y=y, verbose=0)

    # Extract the loss and MAE from the evaluation metrics.
    # loss = metrics[0]
    mae = metrics[1]

    # Print the evaluated metrics, including the loss and MAE rounded to two decimal places.
    # print(f"✅ Model evaluated on the test set: loss {round(loss, 2)}")
    print(f"✅ Model evaluated on the test set: MAE {round(mae, 2)}")


    # Return a dictionary containing the evaluation metrics.
    return metrics
