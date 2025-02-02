# 

from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, r2_score
from utils.data_split import load_and_split_data  # Import the load_and_split_data function
import logging
import os

# Set up the custom logger
log_dir = "logging"
os.makedirs(log_dir, exist_ok=True)  # Ensure the logging directory exists
log_file = os.path.join(log_dir, "evaluation.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Save logs to a file
        logging.StreamHandler()  # Also print logs to the console
    ]
)
logger = logging.getLogger(__name__)

def evaluate_model():
    """Evaluate model and generate reports."""
    logger.info("Starting model evaluation...")
    
    # Define base directory
    base_dir = Path(__file__).parent.parent
    
    # Define path to the model file
    model_file = base_dir / "models" / "best_xgboost_model" / "model.pkl"
    
    # Load model from the specified path
    logger.info(f"Loading model from: {model_file}")
    
    try:
        with open(model_file, "rb") as f:
            loaded_model = pickle.load(f)  # Load the model using pickle
        logger.info(f"Model loaded successfully from: {model_file}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Define paths to the training and test data
    train_file = base_dir / "data" / "processed" / "train.csv"
    test_file = base_dir / "data" / "processed" / "test.csv"
    
    # Load and split the data
    logger.info(f"Loading training data from: {train_file}")
    logger.info(f"Loading test data from: {test_file}")
    X_train, y_train, X_test, y_test = load_and_split_data(train_file, test_file)
    logger.info("Data loaded and split into training and test sets.")
    
    # Generate predictions on the test set
    logger.info("Generating predictions on the test set...")
    preds = loaded_model.predict(X_test)  # Use the loaded model for predictions
    
    # Calculate metrics
    logger.info("Calculating evaluation metrics...")
    metrics = {
        "rmse": mean_squared_error(y_test, preds, squared=False),  # RMSE
        "r2": r2_score(y_test, preds)  # R² score
    }
    logger.info(f"Evaluation metrics: RMSE = {metrics['rmse']}, R^2 = {metrics['r2']}")
    
    # Save metrics in the artifacts directory
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)  # Ensure the artifacts directory exists
    metrics_path = artifacts_dir / "metrics.json"
    pd.DataFrame([metrics]).to_json(metrics_path, orient="records")
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # SHAP analysis
    logger.info("Performing SHAP analysis...")
    try:
        # Create an explainer for SHAP analysis
        explainer = shap.Explainer(loaded_model)
        
        # Ensure SHAP gets feature names
        shap_values = explainer(X_test)
        
        # Save SHAP summary plot in the artifacts directory
        shap_plot_path = artifacts_dir / "shap_summary.png"
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to: {shap_plot_path}")
    
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")

    logger.info("Model evaluation completed.")

if __name__ == "__main__":
    evaluate_model()
