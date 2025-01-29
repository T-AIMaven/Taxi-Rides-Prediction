import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environmental variables
load_dotenv()

GCP_PROJECT=os.getenv('GCP_PROJECT')
BQ_DATASET=os.getenv('BQ_DATASET')
TABLE=os.getenv('TABLE')
BUCKET_NAME = os.getenv('BUCKET_NAME')

MODEL_TARGET = os.getenv('MODEL_TARGET')
DATA_SIZE=os.getenv('DATA_SIZE')
RANDOM_STATE=os.getenv('RANDOM_STATE')
LEARNING_RATE=os.getenv('LEARNING_RATE')
BATCH_SIZE=os.getenv('BATCH_SIZE')

PREFECT_FLOW_NAME=os.getenv('PREFECT_FLOW_NAME')
MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME')



LOCAL_DATA_PATH = Path('~').joinpath("OneDrive", "11_MLOps", "taxifare", "data").expanduser()
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "OneDrive", "11_MLOps",  "taxifare", "training_outputs")
MODEL_DIRECTORY = os.path.join(os.path.expanduser('~'), "OneDrive", "11_MLOps",  "taxifare", "training_outputs", "models")  


# MLFLOW_PATH = "./mlruns/1/9b7ac5169f57453aba3ee28c31480045/artifacts/model"

COLUMN_NAMES_RAW = ['fare',
                    'trip_start_timestamp', 
                    'pickup_longitude', 
                    'pickup_latitude', 
                    'dropoff_longitude', 
                    'dropoff_latitude', 
                    'passenger_count']

DTYPES_RAW = {"fare": "float32",
              "trip_start_timestamp": "datetime64[ns, UTC]",
              "pickup_longitude": "float32",
              "pickup_latitude": "float32",
              "dropoff_longitude": "float32",
              "dropoff_latitude": "float32"}

DTYPES_PROCESSED = np.float32