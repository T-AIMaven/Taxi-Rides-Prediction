import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path
import numpy as np
from scipy import stats

from src.params import *



def get_data_with_cache(gcp_project:str,
                        query:str,
                        cache_path:Path,
                        data_has_header=True) -> pd.DataFrame:
    """
    Retrieve `query` data from Big Query, or from `cache_path` if file exists.
    Store at `cache_path` if retrieved from Big Query for future re-use.
    """
    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)

    else:
        print(Fore.BLUE + "\nLoad data from Querying Big Query server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    # Compress raw_data by setting types to DTYPES_RAW

    df = df.astype(DTYPES_RAW)


    # Drop NaN, duplicates and 0 values in the lat/long
    print(f"Nr of rows before cleaning {df.shape[0]}")

    df = df.drop_duplicates()
    df = df.dropna(how='any', axis=0)
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0) | (df.pickup_latitude != 0) | (df.pickup_longitude != 0)]

    print(f"Nr of rows before cleaning {df.shape[0]}")


    # Compute z-score to remove outliers
    data = np.array(df['fare'])

    # Calculate the z-scores for the data
    z_scores = stats.zscore(data)

    # Define a z-score threshold for identifying outliers
    z_score_threshold = 2  # You can adjust this threshold as needed

    # Identify outliers threshold 
    df_filtered = df[np.abs(stats.zscore(df['fare'])) <= z_score_threshold]

    print("Outliers threshold:", df_filtered.fare.max(), "USD")

    # Remove outliers
    df = df[(df.fare > 0) & (df.fare < df_filtered.fare.max())]

    print(f"Outliers removed. New shape: {df.shape}")

    print("✅ Data cleaned")

    return df



def load_data_to_bq(data: pd.DataFrame,
                    gcp_project:str,
                    bq_dataset:str,
                    table: str,
                    truncate: bool) -> None:
    
    """
    - Save dataframe to bigquery
    - Empty the table beforehands if `truncate` is True, append otherwise.
    """


    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to bigquery {full_table_name}...:" + Style.RESET_ALL)


    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")