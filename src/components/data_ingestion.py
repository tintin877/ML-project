import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging

# Configuration class for paths
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

# Data Ingestion Class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads the raw dataset, splits it into train and test sets,
        and saves these as CSV files in the 'artifacts' folder.
        """
        logging.info("Starting the data ingestion process.")

        try:
            # Read dataset
            logging.info("Reading the dataset from CSV file.")
            df = pd.read_csv("notebook/data2/stud.csv")
            logging.info("Dataset loaded successfully into a DataFrame.")

            # Ensure the 'artifacts' directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            logging.info("Created 'artifacts' directory if it didn't already exist.")

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw dataset saved to artifacts/data.csv.")

            # Split the data into train and test sets
            logging.info("Splitting the dataset into train and test sets.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets to CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f"Train dataset saved to {self.ingestion_config.train_data_path}.")
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Test dataset saved to {self.ingestion_config.test_data_path}.")

            logging.info("Data ingestion process completed successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.error("An error occurred during the data ingestion process.")
            raise CustomException(e, sys)


# Run the script
if __name__ == "__main__":
    # Initialize data ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Import data transformation
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer

    # Data transformation step
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Model training step
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
