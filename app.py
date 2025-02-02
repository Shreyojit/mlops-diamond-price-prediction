# import logging
# from flask import Flask, request, render_template, jsonify
# import joblib
# import pandas as pd
# import numpy as np
# import os

# # Configure custom logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)

# app = Flask(__name__)

# # Load the trained model
# model_path = os.path.join('models', 'best_xgboost_model', 'model.pkl')
# try:
#     model = joblib.load(model_path)
#     logger.info(f"Model loaded successfully from: {model_path}")
# except FileNotFoundError as e:
#     logger.error(f"Model file not found: {e}")
#     raise
# except Exception as e:
#     logger.error(f"Error loading model: {e}")
#     raise

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get data from form
#         data = {
#             'carat': float(request.form['carat']),
#             'cut': request.form['cut'],
#             'color': request.form['color'],
#             'clarity': request.form['clarity'],
#             'depth': float(request.form['depth']),
#             'table': float(request.form['table']),
#             'x': float(request.form['x']),
#             'y': float(request.form['y']),
#             'z': float(request.form['z'])
#         }
        
#         # Create DataFrame
#         df = pd.DataFrame([data])

#         # Preprocess categorical features
#         cut_encoder = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
#         color_encoder = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
#         clarity_encoder = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

#         # Map categorical columns to encoded values
#         df['cut'] = df['cut'].map(cut_encoder)
#         df['color'] = df['color'].map(color_encoder)
#         df['clarity'] = df['clarity'].map(clarity_encoder)
        
#         # Ensure no NaN values after mapping
#         if df.isnull().values.any():
#             logger.warning("Data contains NaN values after mapping categorical features. Filling with 0.")
#             df = df.fillna(0)
        
#         # Make prediction
#         prediction = model.predict(df)

#         logger.info(f"Prediction made: {prediction[0]}")

#         return render_template('index.html', 
#                              prediction_text=f'Predicted Diamond Price: ${prediction[0]:.2f}')
    
#     except KeyError as e:
#         logger.error(f"Missing required field: {e}")
#         return jsonify({'error': f'Missing required field: {e}'}), 400
#     except Exception as e:
#         logger.error(f"Error during prediction: {e}")
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)


# import logging
# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd
# import numpy as np
# import os

# # Configure custom logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)

# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)

# app = Flask(__name__)

# # Load the trained model
# model_path = os.path.join('models', 'best_xgboost_model', 'model.pkl')
# try:
#     model = joblib.load(model_path)
#     logger.info(f"Model loaded successfully from: {model_path}")
# except FileNotFoundError as e:
#     logger.error(f"Model file not found: {e}")
#     raise
# except Exception as e:
#     logger.error(f"Error loading model: {e}")
#     raise

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Check if the request is JSON
#         if request.is_json:
#             data = request.get_json()
#         else:
#             return jsonify({"error": "Request must be in JSON format"}), 400
        
#         # Extract values from JSON
#         data_dict = {
#             'carat': float(data.get('carat', 0)),
#             'cut': data.get('cut', ''),
#             'color': data.get('color', ''),
#             'clarity': data.get('clarity', ''),
#             'depth': float(data.get('depth', 0)),
#             'table': float(data.get('table', 0)),
#             'x': float(data.get('x', 0)),
#             'y': float(data.get('y', 0)),
#             'z': float(data.get('z', 0))
#         }
        
#         # Create DataFrame
#         df = pd.DataFrame([data_dict])

#         # Preprocess categorical features
#         cut_encoder = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
#         color_encoder = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
#         clarity_encoder = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

#         # Map categorical columns to encoded values
#         df['cut'] = df['cut'].map(cut_encoder)
#         df['color'] = df['color'].map(color_encoder)
#         df['clarity'] = df['clarity'].map(clarity_encoder)
        
#         # Ensure no NaN values after mapping
#         if df.isnull().values.any():
#             logger.warning("Data contains NaN values after mapping categorical features. Filling with 0.")
#             df = df.fillna(0)
        
#         # Make prediction
#         prediction = model.predict(df)
        
#         # Convert the prediction (which could be a numpy.float32) to a native Python float
#         prediction_value = float(prediction[0])

#         logger.info(f"Prediction made: {prediction_value}")

#         return jsonify({'prediction': prediction_value})
    
#     except KeyError as e:
#         logger.error(f"Missing required field: {e}")
#         return jsonify({'error': f'Missing required field: {e}'}), 400
#     except Exception as e:
#         logger.error(f"Error during prediction: {e}")
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)



import logging
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os

# Configure custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('models', 'best_xgboost_model', 'model.pkl')
try:
    model = joblib.load(model_path)
    logger.info(f"Model loaded successfully from: {model_path}")
except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request is JSON
        data = request.form
        data_dict = {
            'carat': float(data.get('carat', 0)),
            'cut': data.get('cut', ''),
            'color': data.get('color', ''),
            'clarity': data.get('clarity', ''),
            'depth': float(data.get('depth', 0)),
            'table': float(data.get('table', 0)),
            'x': float(data.get('x', 0)),
            'y': float(data.get('y', 0)),
            'z': float(data.get('z', 0))
        }
        
        # Create DataFrame
        df = pd.DataFrame([data_dict])

        # Preprocess categorical features
        cut_encoder = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
        color_encoder = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
        clarity_encoder = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

        # Map categorical columns to encoded values
        df['cut'] = df['cut'].map(cut_encoder)
        df['color'] = df['color'].map(color_encoder)
        df['clarity'] = df['clarity'].map(clarity_encoder)
        
        # Ensure no NaN values after mapping
        if df.isnull().values.any():
            logger.warning("Data contains NaN values after mapping categorical features. Filling with 0.")
            df = df.fillna(0)
        
        # Make prediction
        prediction = model.predict(df)
        
        # Convert the prediction (which could be a numpy.float32) to a native Python float
        prediction_value = float(prediction[0])

        logger.info(f"Prediction made: {prediction_value}")

        return render_template('index.html', prediction=prediction_value)
    
    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return render_template('index.html', error=f'Missing required field: {e}')
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return render_template('index.html', error=str(e))


