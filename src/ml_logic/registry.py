import glob
import os
import time
import pickle
import mlflow
from mlflow.tracking import MlflowClient

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

from src.params import *

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.keras"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.keras"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.keras")
    model.save(model_path)

    print("✅ Model saved locally")

    # Register a model 
    # model_path_mlruns = f"{MLFLOW_PATH}"  # Replace with the actual path to your model in params.py
    # model_uri = mlflow.register_model(model_path_mlruns, "model-test")
    # print(f"Registered model with URI: {model_uri}")


    if MODEL_TARGET == "gcs":

        model_files = [filename for filename in os.listdir(MODEL_DIRECTORY) if filename.endswith('.keras')]


        # Ensure the list is not empty
        if model_files:
            # Find the most recent model file based on the file modification time
            most_recent_model_path_on_disk = max(model_files, key=os.path.getmtime)
            model_filename = os.path.basename(most_recent_model_path_on_disk)

            print(f"Model filename: {model_filename}")

            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(f"models/{model_filename}")
            blob.upload_from_filename(most_recent_model_path_on_disk)
            
            print("✅ Model saved to gcs")


        else:
            print("❌ No model files found in the directory.")
       
        return None


    return model, model_path


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  
    Return None (but do not Raise) if no model found

    """

    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)
        # Get latest model version name by timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        print(f"{local_model_directory}")
        local_model_paths = glob.glob(f"{local_model_directory}/*")
        print(f"{local_model_paths}")
        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]
        lastest_model = keras.models.load_model(most_recent_model_path_on_disk)
        print("✅ Latest model loaded from local disk")

        return lastest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add breakpoint if you need!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))
        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)
            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        
        except:
            print(f"\n❌ No model found on GCS bucket {BUCKET_NAME}")
            return None


# Model Transition Function
# The application of the function is in main.py
def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from current_stage stage to new_stage and archive the existing model in new_stage
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()

    version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"✅ Model {MLFLOW_MODEL_NAME} version {version[0].version} transitioned from {current_stage} to {new_stage}")

    return None


def mlflow_run(func):
    """Generic function to log params and results to mlflow along with tensorflow autologging

    Args:
        func (function): Function you want to run within mlflow run
        params (dict, optional): Params to add to the run in mlflow. Defaults to None.
        context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        # tracking_uri = mlflow.tracking.get_tracking_uri()
        # print("Current Tracking URI:", tracking_uri)
        mlflow.set_experiment(experiment_name="target_mlflow")
        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            mlflow.set_tags({"model_type": "tensorflow",
                            "dataset": "chicago taxi trips"})
            

            results = func(*args, **kwargs)
        print("✅ Mlflow run autolog done! 💪")
        return results
    return wrapper