import joblib
from pathlib import Path

import pandas as pd
from sklearn.linear_model import ElasticNet

def train(train_data_file_path: Path, test_data_file_path: Path,
          target_column: str, alpha: float, l1_ratio: float, model_path: Path):
    
      train_data = pd.read_csv(train_data_file_path)
      test_data = pd.read_csv(test_data_file_path)

      train_X = train_data.drop([target_column], axis=1)
      test_X = train_data.drop([target_column], axis=1)
      train_y = train_data[target_column]
      test_y = test_data[target_column]

      model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
      model.fit(train_X, train_y)
      # Save model
      joblib.dump(model, model_path)