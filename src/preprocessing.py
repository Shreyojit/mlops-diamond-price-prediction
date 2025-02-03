import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configure custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def preprocess_data():
    """Preprocess diamond price dataset, split into train/test sets, and save copies in both artifacts and processed directories."""
    try:
        # Define base directory (relative to the project root)
        base_dir = Path(__file__).resolve().parent.parent
        logger.info(f"Base directory: {base_dir}")
        
        # Ensure directories exist
        processed_dir = base_dir / 'data' / 'processed'
        processed_dir.mkdir(parents=True, exist_ok=True)

        artifacts_dir = base_dir / 'artifacts'
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Directories ensured: {processed_dir}, {artifacts_dir}")
        
        # Load raw data
        raw_data_path = base_dir / 'data' / 'raw' / 'diamonds.csv'
        logger.info(f"Loading data from: {raw_data_path}")
        data = pd.read_csv(raw_data_path)

        # Remove unnamed columns if they exist
        data = data.drop(["Unnamed: 0"], axis=1, errors='ignore')
        logger.info("Removed unnamed columns from the dataset.")

        # Remove invalid entries
        data = data[(data["x"] > 0) & (data["y"] > 0) & (data["z"] > 0)]
        data = data[(data["depth"] < 75) & (data["depth"] > 45)]
        data = data[(data["table"] < 80) & (data["table"] > 40)]
        data = data[(data["x"] < 30) & (data["y"] < 30) & (data["z"] < 30) & (data["z"] > 2)]
        
        # Perform preprocessing (drop missing values)
        data = data.dropna()
        logger.info("Dropped missing values.")

        # Label encoding for categorical columns
        for col in ['cut', 'color', 'clarity']:
            data[col] = LabelEncoder().fit_transform(data[col])
        
        logger.info("Applied label encoding.")

        # Convert numeric columns to float64
        numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
        data[numeric_columns] = data[numeric_columns].astype('float64')

        # Splitting data into train and test sets
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        logger.info(f"Data split: Train ({len(train_data)}), Test ({len(test_data)})")

        # Define file paths
        train_artifact = artifacts_dir / 'train.csv'
        test_artifact = artifacts_dir / 'test.csv'

        train_processed = processed_dir / 'train.csv'
        test_processed = processed_dir / 'test.csv'

        # Save data
        train_data.to_csv(train_artifact, index=False)
        test_data.to_csv(test_artifact, index=False)
        train_data.to_csv(train_processed, index=False)
        test_data.to_csv(test_processed, index=False)

        logger.info(f"Data saved to artifacts: {train_artifact}, {test_artifact}")
        logger.info(f"Data copied to processed: {train_processed}, {test_processed}")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting data preprocessing...")
    preprocess_data()
    logger.info("Data preprocessing completed successfully.")
