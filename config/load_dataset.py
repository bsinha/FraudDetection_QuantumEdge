import os
import pandas as pd
from kaggle.api.kaggle_api import KaggleApi

# Replace with your Kaggle username and API token
#kaggle_username = "bipulnsinha"
#kaggle_api_token = "a334cf9de790dac03bba55043c928375"

# Define file path and Kaggle dataset information
csv_file = "creditcardfraud.zip"
#dataset_owner = "dataset_owner"  # Replace with owner of the Kaggle dataset
#dataset_name = "dataset_name"  # Replace with the actual dataset name

def download_kaggle_dataset(file_path,  dataset_name):
    """Downloads a dataset from Kaggle if it doesn't exist locally.

    Args:
        file_path (str): Path to the local CSV file.
        dataset_owner (str): Username of the Kaggle dataset owner.
        dataset_name (str): Name of the Kaggle dataset.
    """

    if not os.path.exists(file_path):
        # Configure Kaggle API (replace with your credentials)
        api = KaggleApi()
        
        print(api)
        #api.authenticate(username=kaggle_username, key=kaggle_api_token)
        #api.dataset_download_file(dataset_owner, dataset_name)
        api.datasets_download_file(file_name=csv_file, owner_slug="mlg-ulb", dataset_slug="creditcardfraud")
        print(f"Downloaded {dataset_name} from Kaggle.")

def read_csv_file(file_path):
    """Reads data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: The DataFrame containing the CSV data.
    """

    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"CSV file {file_path} not found. Downloading from Kaggle...")
        download_kaggle_dataset(file_path, csv_file)
        return read_csv_file(file_path)  # Recursive call to read after download

# Download and read the CSV file
#data = read_csv_file(csv_file)

# Now you can use the data in your analysis (replace with your logic)
#print(data.head())  # Print the first few rows

