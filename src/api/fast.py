import pandas as pd
from fastapi import FastAPI
from src. ml_logic.preprocessor import preprocess_features
from src.ml_logic.registry import load_model


# Create an instance of FastAPI
app = FastAPI()

# Store the model in an `app.state.model` global variable accessible across all routes!
app.state.model = load_model()

@app.get("/predict")
def prediction():
    
    """
    Make a prediction using the latest trained model
    """
   
    # Make predictions on the validation and test datasets
    X_pred = pd.DataFrame(dict(
        trip_start_timestamp=[pd.Timestamp("2023-09-01 00:00:00+00:00", tz='UTC')],
        pickup_longitude=[-87.632744],
        pickup_latitude=[41.880993],
        dropoff_longitude=[-87.637848],
        dropoff_latitude=[41.893215]
    ))


    model = app.state.model
        
    
    # Preprocess features
    preprocessor = preprocess_features()
    X_train_processed = preprocessor.fit_transform(X_pred)

    # Predict
    # Convert in a JSON-serializable value
    y_test_pred = float(model.predict(X_train_processed)[0][0])
         

    return {"Prediction (USD)": round(y_test_pred, 2)}  # Return the prediction as a JSON object



# Define a route and a function to handle requests to that route
@app.get("/")
def read_root():
    return {"Message": "Welcome to the Chicago Taxi prediction App!"}


## For the website. Inputs come from there
# from fastapi import Query

# @app.get("/predict")
# def prediction(
#     trip_start_timestamp: str = Query(..., description="Start timestamp in ISO 8601 format"),
#     pickup_longitude: float = Query(..., description="Pickup longitude"),
#     pickup_latitude: float = Query(..., description="Pickup latitude"),
#     dropoff_longitude: float = Query(..., description="Dropoff longitude"),
#     dropoff_latitude: float = Query(..., description="Dropoff latitude")
# ):
#     """
#     Make a prediction using the latest trained model
#     """

#     # Convert input data to the expected format
#     X_pred = pd.DataFrame(dict(
#         trip_start_timestamp=[pd.Timestamp(trip_start_timestamp, tz='UTC')],
#         pickup_longitude=[pickup_longitude],
#         pickup_latitude=[pickup_latitude],
#         dropoff_longitude=[dropoff_longitude],
#         dropoff_latitude=[dropoff_latitude]
#     ))

#     model = app.state.model
#     assert model is not None

#     # Preprocess features
#     preprocessor = preprocess_features()
#     X_train_processed = preprocessor.fit_transform(X_pred)

#     # Predict
#     y_test_pred = float(model.predict(X_train_processed)[0][0])

#     return {"Prediction (USD)": round(y_test_pred, 2)}