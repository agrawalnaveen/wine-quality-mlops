import logging
from urllib import request
from zipfile import ZipFile
from pathlib import Path


import great_expectations as gx
import pandas as pd
from sklearn.model_selection import train_test_split

# Create a logger for the module
logger = logging.getLogger(__name__)

def download_data(data_url: str, zip_data_file_path: str, unzip_data_path: str) -> None:
    try:
        filename, headers = request.urlretrieve(data_url, zip_data_file_path)
        logger.info(f'Downloding file: {filename}')
    except Exception as e:
        print("Unable to download data")
        raise e
    # Extract the downloaded zip file
    with ZipFile(zip_data_file_path, 'r') as zip_file:
        zip_file.extractall(unzip_data_path)

def validate_data(project_root_dir: str,
                  checkpoint_name: str,
                  validation_result_file_path: str) -> bool:
    context = gx.get_context()
    checkpoint = context.get_checkpoint(name=checkpoint_name)
    checkpoint_result = checkpoint.run()
    result = checkpoint_result.list_validation_results()[0]["success"]
    with open(validation_result_file_path, 'w') as f:
        f.write(str(result))

def transform_data(raw_data_file_path: Path,
                   test_size: float, transformed_data_dir: Path) -> None:
    data = pd.read_csv(raw_data_file_path, delimiter=';')
    train, test = train_test_split(data, test_size=test_size)
    train.to_csv(transformed_data_dir / 'train.csv', index=False)
    test.to_csv(transformed_data_dir / 'test.csv', index=False)