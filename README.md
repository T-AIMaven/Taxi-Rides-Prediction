# MLOps Project Chicago Taxi Prices Prediction 🚕

<p>
    <img src="/images/taxi.jpg"/>
    </p>

This is a personal MLOps project based on a [BigQuery](https://console.cloud.google.com/marketplace/product/city-of-chicago-public-data/chicago-taxi-trips?project=taxifare-mlops) dataset for taxi ride prices in Chicago. Below you can find some instructions to understand the project content. Feel free to ⭐ and clone this repo 😉


## Tech Stack

![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23d9ead3.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-0194E2.svg?style=for-the-badge&logo=MLflow&logoColor=white)
![Prefect](https://img.shields.io/badge/Prefect-024DFD.svg?style=for-the-badge&logo=Prefect&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Project Structure

The project has been structured with the following folders and files:

- `images:` images from results
- `notebooks:` EDA and Modelling performed at the beginning of the project to establish a baseline
- `src:` source code. It is divided into:
    - `api`: FastApi app code
    - `interface`: main workflows
    - `ml_logic`: data/preprocessing/modelling functions

- `requirements.txt:` project requirements
- `Dockerfile`: docker image for deployment
- `.env.sample`: sample file of environmental variables
- `.env.sample.yaml`: sample file of environmental variables for Dockerfile deployment

## Project Description

The dataset was obtained from BigQuery and contains 200 million rows and various columns from which the following were selected for this project: prices, pick-up and drop-off locations, and timestamps. To prepare the data for modelling, an **Exploratory Data Analysis** was conducted to preprocess time and distance features, and suitable scalers and encoders were chosen for the preprocessing pipeline.

### Fare Distribution

The following two charts show the fare distribution of the rides. As the number of rows is too big an environmental variable (`DATA_SIZE`) was set up to decide how many rows to query. However, the price distribution for the first 1 million rows shows a big concentration in the first 100 USD.

<p>
    <img src="/images/prices_distribution.png"/>
    </p>

<p>
    <img src="/images/prices_distribution2.png"/>
    </p>


 In order to detect outliers, the `z-score` is calculated for each query, so that the outliers are removed depending on the number of rows downloaded. The following chart represents the fare distribution after removing outliers.

<p>
    <img src="/images/clean_prices.png"/>
    </p>

### Distance Distribution

For the distance preprocessing, the first approach was to plot the pickup and drop-off locations on a map and histogram (excluding outliers), to see the distribution.

<p>
    <img src="/images/distance_map.png"/>
    </p>

<p align="center">
    <img src="/images/dist_hist.png"/>
    </p>

It can be seen that the distance distribution is heavily concentrated in the first 10 km to 50 km. The preprocessing approach was to calculate the `Manhattan` and `Haversine` distance for each ride and encode it.

### Time Distribution

For the time preprocessing, the idea was to extract the hour/day/month and separate features and encode them. The hours were previously divided into sine and cosine.

<p align="center">
    <img src="/images/time_features.png"/>
    </p>

Subsequently, a **Neural Network Model** was performed with several Dense, BatchNormalization and Dropout layers. The results showed a MAE of around 3 USD from an average price of 20 USD. However, the price prediction for rides above 10 USD shows a higher accuracy compared to rides up to 10 USD.

<p>
    <img src="/images/prediction.png"/>
    </p>

Afterwards, the models underwent model registry, and deployment using MLflow, Prefect, and FasApi. The Dockerimage was pushed to a Google Container Registry and deployed in Google Cloud Run.


### Modelling

In order to train a model, the file `main.py` in the src/interface folder must be run. This will log the models in MLflow and allow registration and model transition from *None* to *Staging* and *Production* stages. These options can be set up in the file `registry.py` in the src/ml_logic folder. Additionally, the environmental variable `MODEL_TARGET` must be set either to *local* or *gcs*, so that the model is saved either locally or in a GCS Bucket.

<p>
    <img src="/images/mlflowgc.png"/>
    </p>
 
Once a model is saved/registered, the `workflow.py` file in the src/interface folder allows a Prefect workflow to predict new data with the saved model and train a new model with these data to compare the results. If the MAE of the new model is lower, this model can be sent to the production stage and the old model will be archived.

<p>
    <img src="/images/prefect_flow.png"/>
    </p>

To run Prefect and MLflow the following commands must be run in the terminal from the src/interface directory, to see the logs:

- MLFlow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

- Prefect Cloud (with own account):

```bash
prefect cloud login
```

- Prefect (locally): 

    ```bash
    prefect server
    prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
    ```

### Deployment 

Having a model saved and in production, the `fast.py` file can be run to get a prediction. This can be done either locally by running a prediction API, building a Dockerfile, or pushing the Dockerfile to a Docker container in Google Cloud Run to get a service URL.

<p>
    <img src="/images/uvicorn.png"/>
    </p>
 
<p>
    <img src="/images/uvicorn2.png"/>
    </p>


#### Prediction API

To run the prediction API run this from the project root directory and check the results here http://127.0.0.1:8000/predict:

```bash
uvicorn src.api.fast:app --reload
```

#### Dockerimage

To run the Dockerimage build it and check the results here http://127.0.0.1:8000/predict:

```bash
docker build --tag=image .
docker run -it -e PORT=8000 -p 8000:8000 --env-file your/path/to/.env image
```

#### Dockerimage in Google Cloud

To get a service URL, first build the image:

```bash
docker build --tag=image .
```

Then push our image to Google Container Registry:

```bash
docker push image
```

Finally, deploy it and get the Service URL in the terminal to run predictions on your own website. You should get something like this: `Service URL: https://yourimage-jdhsk768sdfd-rt.a.run.app`

```bash
gcloud run deploy --image image --region your-gcp-region --env-vars-file .env.yaml
```
