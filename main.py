import json
import logging
# import logging.config
import yaml
from pathlib import Path

# # Load logging configuration from the YAML file
# with open('logging_config.yaml', 'rt') as f:
#     loging_config = yaml.safe_load(f.read())

# # Configure logging using the loaded configuration
# logging.config.dictConfig(loging_config)

# Load configuration from the YAML file
with open('config.yaml', 'rt') as f:
    config = yaml.safe_load(f.read())
## Config variables for downloading data    
data_url = config["data_ingestion"]["data_url"]
zip_data_file_path = config["data_ingestion"]["zip_data_file_path"]
unzip_data_path = config["data_ingestion"]["unzip_data_path"]

## Config variables for validating data
project_root_dir = config["data_validation"]["project_root_dir"]
checkpoint_name = config["data_validation"]["checkpoint_name"]
validation_result_file_path = config['data_validation']['validation_result_file_path']

## Config variable for data transformation
raw_red_wine_data_file_path: str = config["data_transformation"]["raw_red_wine_data_file_path"]
transformed_data_dir: str = config["data_transformation"]["transformed_data_dir"]
test_size: float = config["data_transformation"]["test_size"]

## Config variables for training model
train_data_file_path = config['train']['train_data_file_path']
test_data_file_path = config['train']['test_data_file_path']
target_column = config['train']['target_column']
alpha = config['train']['alpha']
l1_ratio = config['train']['l1_ratio']
model_path = Path(config['train']['model_path'])

## Config variables for model evaluation
metrics_file_path = config['evaluate']['metrics_file_path']
MLFLOW_TRACKING_URI = config['track']['MLFLOW_TRACKING_URI']
experiment_name = config['track']['experiment_name']

# Import modules
from wine import data, model

# Download data
data.download_data(data_url, zip_data_file_path, unzip_data_path)

# Validate data
data.validate_data(project_root_dir=project_root_dir,
                   checkpoint_name=checkpoint_name,
                   validation_result_file_path=validation_result_file_path)

# Check if data validation was successfull
with open(validation_result_file_path, 'r') as f:
    val_result = f.read()
if val_result == "True":
    data.transform_data(raw_data_file_path=Path(raw_red_wine_data_file_path),
                        test_size=test_size,
                        transformed_data_dir=Path(transformed_data_dir))
else:
    raise Exception("Unsuccessful data validation")

# Create, train, and save model
params = {
    'alpha': alpha,
    'l1_ratio': l1_ratio,
}
model.train(train_data_file_path=train_data_file_path,
            target_column=target_column,
            **params,
            model_path=model_path)

model.evaluate(model_path=model_path,
               test_data_file_path=test_data_file_path,
               target_column=target_column,
               metrics_file_path=metrics_file_path)

with open(metrics_file_path, 'r') as metrics_file:
        metrics = json.load(metrics_file)

model.track_with_mlflow(uri=MLFLOW_TRACKING_URI,experiment_name=experiment_name,
                        params=params, metrics=metrics)


# # You can also use the root logger if needed
# logging.info("Debug message from the main script")
