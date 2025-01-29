import numpy as np
import pandas as pd

from colorama import Fore, Style

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, MinMaxScaler, RobustScaler

from src.ml_logic.encoders import transform_time_features, transform_lonlat_features


def preprocess_features():
    
    def create_sklearn_preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of fixed shape (_, 65).

        Stateless operation: "fit_transform()" equals "transform()".
        """

        # DISTANCE PIPE

        lonlat_features = ["pickup_latitude", 
                           "pickup_longitude", 
                           "dropoff_latitude", 
                           "dropoff_longitude"]

        distance_pipe = make_pipeline(
            FunctionTransformer(transform_lonlat_features),
            RobustScaler()  
        )



        # TIME PIPE       
        time_categories = [np.arange(0, 7, 1),   # days of the week from 0 to 6
                           np.arange(1, 13, 1)]  # months of the year from 1 to 12
        
        time_pipe = make_pipeline(
            FunctionTransformer(transform_time_features),
            make_column_transformer(
                (OneHotEncoder(
                    categories=time_categories,
                    sparse_output=False,
                    handle_unknown="ignore"
                ), [2, 3]),
                (MinMaxScaler(), [4]),  
                remainder="passthrough"
            )
        )



        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer(
            [
                ("time_preproc", time_pipe, ["trip_start_timestamp"]),
                ("dist_preproc", distance_pipe, lonlat_features),
            ],
            n_jobs=-1,
        )

        return final_preprocessor

    
    preprocessor = create_sklearn_preprocessor()


    return preprocessor
