from pathlib import Path
import pandas as pd

def load_and_split_data(train_file, test_file):
    """
    Loads the train and test CSV files, and splits them into features (X) and target (y).
    Args:
    - train_file: Path to the training CSV file.
    - test_file: Path to the test CSV file.
    
    Returns:
    - X_train, y_train: Features and target for the training data.
    - X_test, y_test: Features and target for the test data.
    """
    # Load the training and test data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # Assuming your target column is 'price' and all other columns are features
    X_train = train_data.drop(columns=["price"])
    y_train = train_data["price"]
    
    X_test = test_data.drop(columns=["price"])
    y_test = test_data["price"]
    
    return X_train, y_train, X_test, y_test

# Construct the path to the 'data' folder, which is outside 'src'
data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"  # Going two levels up from 'src/utils'

train_file = data_dir / "train.csv"
test_file = data_dir / "test.csv"

print(f"Loading training data from: {train_file}")
print(f"Loading test data from: {test_file}")

X_train, y_train, X_test, y_test = load_and_split_data(train_file, test_file)
