from prefect import task, flow
from src.interface.main import preprocess, evaluate_model, train
from src.ml_logic.registry import load_model, mlflow_transition_model
from sklearn.model_selection import train_test_split
from src.ml_logic.preprocessor import preprocess_features
from src.params import *


# Define tasks with docstrings

@task(name="Preprocessing new data")
def preprocess_new_data():
    """
    Preprocesses new data and returns the processed data.
    """
    return preprocess()

@task(name="Evaluating current model in production")
def evaluate_production_model(model, X_test_processed, y_test):
    """
    Evaluates the current model on production data and returns the evaluation results.
    """
    return evaluate_model(model, X_test_processed, y_test)

@task(name="Training new model")
def re_train(X, y, random_state, learning_rate, batch_size):
    """
    Trains a new model with the provided data and hyperparameters and returns the trained model.
    """
    return train(X, y, random_state, learning_rate, batch_size)

@task(name="Evaluating new trained model")
def evaluate_new_model(model, X_test_processed, y_test):
    """
    Evaluates a newly trained model on test data and returns the evaluation results.
    """
    return evaluate_model(model, X_test_processed, y_test)

@task(name="Transition model")
def transition_model(current_stage: str, new_stage: str):
    """
    Transitions the model from the current stage to a new stage using MLflow.
    """
    return mlflow_transition_model(current_stage, new_stage)

# Define the Prefect flow

@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Main Prefect flow for training and evaluating models.
    """

    # Getting new data
    X, y = preprocess_new_data()


    ###     EVALUATING CURRENT MODEL IN PRODUCTION    ###

    old_model = load_model()

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

    print("\n✅ Splitting dataframe")

    # Print the shapes of the resulting DataFrames
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    print("✅ Splitting done \n")

    # Preprocessing
    preprocessor = preprocess_features()

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    # Evaluating
    old_mae = evaluate_production_model(old_model, X_test_processed, y_test)[1]

    print(f"✅ Model evaluated on the old model: MAE {round(old_mae, 2)}")


    ###     EVALUATING NEW MODEL    ###

    new_model, X_train, X_test_processed, y_test = re_train(X, y, RANDOM_STATE, LEARNING_RATE, BATCH_SIZE)
    
    new_mae = evaluate_new_model(new_model, X_test_processed, y_test)[1]
     
    print(f"✅ Model evaluated on the new model: MAE {round(new_mae, 2)}")

    ###     MODEL TRANSITION    ###

    if old_mae > new_mae:
        print(f"↗️ New model replacing old in production with MAE: {round(new_mae, 2)}. The Old MAE was: {round(old_mae, 2)}")

        transition_model("Staging", "Production")



if __name__ == "__main__":
    try:
        train_flow()
    except Exception as err:
        print(f"An error occurred: {err}")
