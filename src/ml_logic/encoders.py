import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environmental variables


def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    
    # Find the oldest time in the DataFrame to use as a reference point
    reference_time = X["trip_start_timestamp"].min()
    
    # Calculate the time difference in days from the oldest time
    timedelta = (X["trip_start_timestamp"] - reference_time) / pd.Timedelta(1, 'D')
    
    # Extract additional time-related features from "trip_start_timestamp"
    pickup_dt = X["trip_start_timestamp"].dt.tz_convert("America/Chicago").dt
    dow = pickup_dt.weekday
    hour = pickup_dt.hour
    month = pickup_dt.month
    
    # Encode the hour of the day using sine and cosine to capture cyclic patterns
    hour_sin = np.sin(2 * math.pi / 24 * hour)
    hour_cos = np.cos(2 * math.pi / 24 * hour)
    
    # Stack the extracted features horizontally into a NumPy array
    feature_array = np.stack([hour_sin, hour_cos, dow, month, timedelta], axis=1)
    
    return feature_array


def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:

    lonlat_features = ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]

    def distances_vectorized(df: pd.DataFrame, start_lat: str, start_lon: str, end_lat: str, end_lon: str) -> dict:
        """
        Calculate the haverzine and manhattan distance between two points on the earth (specified in decimal degrees).
        Vectorized version for pandas df
        Computes distance in kms
        """
        earth_radius = 6371

        lat_1_rad, lon_1_rad = np.radians(df[start_lat]), np.radians(df[start_lon])
        lat_2_rad, lon_2_rad = np.radians(df[end_lat]), np.radians(df[end_lon])

        dlon_rad = lon_2_rad - lon_1_rad
        dlat_rad = lat_2_rad - lat_1_rad

        manhattan_rad = np.abs(dlon_rad) + np.abs(dlat_rad)
        manhattan_km = manhattan_rad * earth_radius

        a = (np.sin(dlat_rad / 2.0)**2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon_rad / 2.0)**2)
        haversine_rad = 2 * np.arcsin(np.sqrt(a))
        haversine_km = haversine_rad * earth_radius

        return dict(
            haversize=haversine_km,
            manhattan=manhattan_km)

    result = pd.DataFrame(distances_vectorized(X, *lonlat_features))

    return result
