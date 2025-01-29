import os
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

from colorama import Fore, Style

from sklearn.model_selection import train_test_split

from src.ml_logic.registry import save_model, save_results, mlflow_run, load_model
from src.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model 
from src.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from src. ml_logic.preprocessor import preprocess_features
from src.params import *


# Preprocess the data
# @mlflow_run
def preprocess() -> None:

    """
    Preprocess and load data into BigQuery.

    This function retrieves data from a specified BigQuery table, preprocesses it,
    and loads the processed data back into a new BigQuery table. It follows these steps:
    
    1. Retrieve data from BigQuery or cache if available.
    2. Clean the retrieved data.
    3. Split the data into features (X) and the target variable (y).
    4. Preprocess the feature data.
    5. Create a DataFrame with timestamp, processed features, and the target variable.
    6. Load the processed data into a new BigQuery table.

    Note: The specific details of data retrieval, cleaning, preprocessing, and loading are
    assumed to be implemented in external functions (get_data_with_cache, clean_data,
    preprocess_features, load_data_to_bq), which need to be defined elsewhere.

    Returns:
        None

    """

    # Print a colored message to indicate the start of the "preprocess_and_train" use case
    print(Fore.MAGENTA + "\n ⭐️ Cleaning and Preprocess" + Style.RESET_ALL)

    # Define the columns to be selected from the BigQuery table
    selected_columns = ['trip_start_timestamp', 'fare',  
                        'pickup_latitude', 'pickup_longitude', 
                        'dropoff_latitude', 'dropoff_longitude']

    # Construct a SQL query to retrieve data from BigQuery
    query = f"""
        SELECT {', '.join(selected_columns)}
        FROM {GCP_PROJECT}.{BQ_DATASET}.{TABLE}
        ORDER BY trip_start_timestamp DESC
        LIMIT {DATA_SIZE}
        """

    # Define a path for caching the query results
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{DATA_SIZE}.csv")

    # Retrieve data from BigQuery or from a cache file if it already exists
    data_query = get_data_with_cache(query=query,
                                     gcp_project=GCP_PROJECT,
                                     cache_path=data_query_cache_path,
                                     data_has_header=True)

    # Clean the retrieved data
    data_clean = clean_data(data_query)


    # Shuffle the data
    data_clean = data_clean.sample(frac=1, random_state=RANDOM_STATE)


    # Split dataset
    X = data_clean.drop("fare", axis=1)
    y = data_clean[["fare"]]


    # Load the processed data into a BigQuery table
    load_data_to_bq(data_clean,
                    gcp_project=GCP_PROJECT,
                    bq_dataset=BQ_DATASET,
                    table=f'processed_{DATA_SIZE}',
                    truncate=True)

    # Print a message to indicate the completion of the preprocessing step
    print("✅ Cleaning and ingestion done \n")

    return X, y


@mlflow_run
def train(X, 
          y,
          random_state,
          learning_rate,
          batch_size):
    
    print(Fore.MAGENTA + "\n ⭐️ Training and Evaluation" + Style.RESET_ALL)
    
  
    # Split the data into train, validation, and test sets (80% train, 15% validation, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE)

    print("\n✅ Splitting dataframe")

    # Print the shapes of the resulting DataFrame
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    print("✅ Splitting done \n")
    
    
    # Get preprocessor
    preprocessor = preprocess_features()


    # Preprocess the feature data
    X_train_processed = preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    


    # Initialize the model
    tf.random.set_seed(random_state)

    # Train model using `model.py`
    # model = load_model()
    # if model is None:
    model = initialize_model(input_shape=X_train_processed.shape[1:])

    # Compile the model
    model = compile_model(model, learning_rate)

    # Train the model
    model, history =  train_model(model,
                                  X_train_processed, 
                                  y_train,
                                  batch_size,
                                  validation_data=(X_val_processed, y_val))


    # Compute the validation metric (min val mae of the holdout set)
    val_mae = np.min(history.history['val_mae'])

    # Save trained model
    params = dict(learning_rate=learning_rate,
                  training_set_size=DATA_SIZE,
                  batch_size=batch_size)

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)


    
    ## The latest model should be moved to staging (Options: "None, "Staging", "Production")
    ## The function is in registry.py 
    # mlflow_transition_model("None", "Production")

   
    print("✅ Preprocess and training done!💪")

    
    return model, X_train, X_test_processed, y_test


def prediction(model,
               X_train,
               X_pred,
               y_test_pred):
    
    """
    Make a prediction using the latest trained model
    """

    print(Fore.MAGENTA + "\n ⭐️ Prediction" + Style.RESET_ALL)

    
    # Make predictions on the validation and test datasets
    if X_pred is None:
        X_pred = pd.DataFrame(dict(
            trip_start_timestamp=[pd.Timestamp("2023-09-01 00:00:00+00:00", tz='UTC')],
            pickup_longitude=[-87.632744],
            pickup_latitude=[41.880993],
            dropoff_longitude=[-87.637848],
            dropoff_latitude=[41.893215]
        ))


    model = load_model()
   
    # Preprocess features
    preprocessor = preprocess_features()
    X_train_processed = preprocessor.fit_transform(X_train)
    X__pred_processed = preprocessor.transform(X_pred)

    # Predict
    y_test_pred = model.predict(X__pred_processed)[0][0]

    print("\n✅ Predicted price: ", round(float(y_test_pred), 2), "USD")
          

    return y_test_pred



if __name__ == '__main__':
    X, y = preprocess()

    model, X_train, X_test_processed, y_test = train(X,
                                                     y,
                                                     RANDOM_STATE,
                                                     LEARNING_RATE,
                                                     BATCH_SIZE)

    evaluate_model(model, 
                   X_test_processed, 
                   y_test)
    
    # Add a prediction ride in None
    prediction(model,
               X_train,
               None,
               y_test)
    