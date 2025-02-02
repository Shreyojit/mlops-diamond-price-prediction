# import os
# import shutil
# import yaml
# import mlflow
# import mlflow.xgboost
# from mlflow.models.signature import infer_signature
# import xgboost as xgb
# from src.utils.data_split import load_and_split_data  # Import the load_and_split_data function
# from sklearn.metrics import mean_squared_error, r2_score
# import shap
# import dagshub  # Import DagsHub for tracking
# import logging
# import numpy as np
# import matplotlib.pyplot as plt  # Import matplotlib for saving plots

# # Get the absolute path to the config.yaml file (relative to the current script's location)
# config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.yaml')

# # Load configurations from config.yaml
# with open(config_path, 'r') as f:
#     config = yaml.safe_load(f)

# # Create 'logging' folder if it doesn't exist
# log_dir = "logging"
# os.makedirs(log_dir, exist_ok=True)

# # Set up the custom logger
# log_file = os.path.join(log_dir, "model_training.log")
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler(log_file),  # Save logs to a file
#         logging.StreamHandler()  # Also print logs to the console
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize DagsHub MLflow tracking
# dagshub.init(
#     repo_owner="shreyojitdas95",
#     repo_name="mlops-diamond-price-prediction",
#     mlflow=True
# )

# def train_model():
#     logger.info("Starting the model training process...")
    
#     # Define the path to the training and test data
#     train_file = os.path.join("data", "processed", "train.csv")
#     test_file = os.path.join("data", "processed", "test.csv")
    
#     logger.info(f"Loading training data from: {train_file}")
#     logger.info(f"Loading test data from: {test_file}")
    
#     # Load and split the data
#     X_train, y_train, X_test, y_test = load_and_split_data(train_file, test_file)
#     logger.info(f"Data loaded and split into training and test sets.")
    
#     # Initialize the XGBoost model
#     logger.info("Initializing the XGBoost model...")
#     pipeline_xgb = xgb.XGBRegressor(
#         objective='reg:squarederror',
#         n_estimators=config["model"]["n_estimators"],
#         learning_rate=config["model"]["learning_rate"],
#         max_depth=config["model"]["max_depth"],
#         random_state=config["model"]["random_state"]
#     )
    
#     # Initialize MLflow tracking
#     mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    
#     with mlflow.start_run() as run:
#         logger.info("Training the XGBoost model...")
#         pipeline_xgb.fit(X_train, y_train)
#         logger.info("Model training completed.")
        
#         # Make predictions and calculate metrics
#         y_pred = pipeline_xgb.predict(X_test)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)
#         logger.info(f"Model evaluation metrics: RMSE = {rmse}, R^2 = {r2}")
        
#         # Log parameters and metrics
#         mlflow.log_params(pipeline_xgb.get_params())
#         mlflow.log_metrics({"rmse": rmse, "r2": r2})
#         mlflow.set_tag("model_version", config["model"]["version"])
        
#         # Infer the signature and log the model in MLflow
#         signature = infer_signature(X_train, y_pred)
#         mlflow.xgboost.log_model(pipeline_xgb, "model", signature=signature)
        
#         # Generate SHAP summary plot
#         logger.info("Generating SHAP summary plot...")
#         explainer = shap.Explainer(pipeline_xgb)
#         shap_values = explainer(X_train)
#         shap.summary_plot(shap_values, X_train, show=False)
#         plt.savefig("shap_summary.png", bbox_inches='tight')
#         plt.close()
#         mlflow.log_artifact("shap_summary.png")
        
#         # Check if the current model is the best based on R² score
#         if r2 > config["model"]["r2_threshold"]:
#             logger.info("New best model found! Saving and clearing previous models...")
            
#             # Clear the local directory for the best model
#             best_model_dir = os.path.join("models", "best_xgboost_model")
#             if os.path.exists(best_model_dir):
#                 shutil.rmtree(best_model_dir)  # Remove the directory and its contents
#             os.makedirs(best_model_dir, exist_ok=True)  # Recreate the directory
            
#             # Save the new best model locally
#             best_model_path = os.path.join(best_model_dir, "model.pkl")
#             mlflow.xgboost.save_model(pipeline_xgb, best_model_path)
#             logger.info(f"Best model saved locally at: {best_model_path}")
            
#             # Register the model in MLflow Model Registry
#             model_uri = f"runs:/{run.info.run_id}/model"
#             mlflow.register_model(model_uri, "best_xgb_model")
#             logger.info("Model registered in MLflow Model Registry.")
#         else:
#             logger.info("Current model does not meet the performance threshold. Skipping save.")
    
#     logger.info("Model training and logging complete.")

# if __name__ == "__main__":
#     train_model()




import os
import shutil
import yaml
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
import xgboost as xgb
import joblib  # Import joblib for saving model
from utils.data_split import load_and_split_data  # Import the load_and_split_data function
from sklearn.metrics import mean_squared_error, r2_score
import shap
import dagshub  # Import DagsHub for tracking
import logging
import numpy as np
import matplotlib.pyplot as plt  # Import matplotlib for saving plots

# Get the absolute path to the config.yaml file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config.yaml')

# Load configurations from config.yaml
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Create 'logging' folder if it doesn't exist
log_dir = "logging"
os.makedirs(log_dir, exist_ok=True)

# Set up the custom logger
log_file = os.path.join(log_dir, "model_training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Save logs to a file
        logging.StreamHandler()  # Also print logs to the console
    ]
)
logger = logging.getLogger(__name__)

# Initialize DagsHub MLflow tracking
dagshub.init(
    repo_owner="shreyojitdas95",
    repo_name="mlops-diamond-price-prediction",
    mlflow=True
)

def train_model():
    logger.info("Starting the model training process...")
    
    # Define the path to the training and test data
    train_file = os.path.join("data", "processed", "train.csv")
    test_file = os.path.join("data", "processed", "test.csv")
    
    logger.info(f"Loading training data from: {train_file}")
    logger.info(f"Loading test data from: {test_file}")
    
    # Load and split the data
    X_train, y_train, X_test, y_test = load_and_split_data(train_file, test_file)
    logger.info("Data loaded and split into training and test sets.")
    
    # Initialize the XGBoost model
    logger.info("Initializing the XGBoost model...")
    pipeline_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=config["model"]["n_estimators"],
        learning_rate=config["model"]["learning_rate"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"]
    )
    
    # Initialize MLflow tracking
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    
    with mlflow.start_run() as run:
        logger.info("Training the XGBoost model...")
        pipeline_xgb.fit(X_train, y_train)
        logger.info("Model training completed.")
        
        # Make predictions and calculate metrics
        y_pred = pipeline_xgb.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Model evaluation metrics: RMSE = {rmse}, R^2 = {r2}")
        
        # Log parameters and metrics
        mlflow.log_params(pipeline_xgb.get_params())
        mlflow.log_metrics({"rmse": rmse, "r2": r2})
        mlflow.set_tag("model_version", config["model"]["version"])
        
        # Infer the signature and log the model in MLflow
        signature = infer_signature(X_train, y_pred)
        mlflow.xgboost.log_model(pipeline_xgb, "model", signature=signature)
        
        # Generate SHAP summary plot
        logger.info("Generating SHAP summary plot...")
        explainer = shap.Explainer(pipeline_xgb)
        shap_values = explainer(X_train)
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig("shap_summary.png", bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("shap_summary.png")
        
        # Check if the current model is the best based on R² score
        if r2 > config["model"]["r2_threshold"]:
            logger.info("New best model found! Saving and clearing previous models...")
            
            # Clear the local directory for the best model
            best_model_dir = os.path.join("models", "best_xgboost_model")
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)  # Remove the directory and its contents
            os.makedirs(best_model_dir, exist_ok=True)  # Recreate the directory
            
            # Save the new best model locally as a .pkl file
            best_model_path = os.path.join(best_model_dir, "model.pkl")
            joblib.dump(pipeline_xgb, best_model_path)
            logger.info(f"Best model saved locally at: {best_model_path}")
            
            # Register the model in MLflow Model Registry
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, "best_xgb_model")
            logger.info("Model registered in MLflow Model Registry.")
        else:
            logger.info("Current model does not meet the performance threshold. Skipping save.")
    
    logger.info("Model training and logging complete.")

if __name__ == "__main__":
    train_model()
