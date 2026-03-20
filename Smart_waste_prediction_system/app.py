from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model and feature names with error handling
try:
    model = joblib.load('best_food_waste_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    logging.info(f"Model loaded successfully. Features: {len(feature_names)}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None
    feature_names = None

# Weather mapping
weather_mapping = {
    'Sunny': 1,
    'Cloudy': 2,
    'Rainy': 3,
    'Stormy': 4
}

# Day mapping
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

@app.route('/')
def home():
    return render_template('index.html', 
                         days=days,
                         weather_options=weather_mapping.keys())

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html',
                             error="Model not loaded. Please contact administrator.",
                             days=days,
                             weather_options=weather_mapping.keys())
    
    try:
        # Get form data
        day = request.form.get('day', '')
        weather = request.form.get('weather', '')
        festival = int(request.form.get('festival', 0))
        expected_customers = float(request.form.get('expected_customers', 0))
        prev_day_consumption = float(request.form.get('prev_day_consumption', 0))
        prev_week_consumption = float(request.form.get('prev_week_consumption', 0))
        
        # Validate inputs
        if not day or not weather:
            raise ValueError("Please select day and weather")
        
        if expected_customers < 300 or expected_customers > 650:
            raise ValueError("Expected customers should be between 300 and 650")
        
        # Create feature dictionary
        input_data = {col: 0 for col in feature_names}
        
        # Fill in the values
        input_data['Expected_Customers'] = expected_customers
        input_data['Previous_Day_Consumption'] = prev_day_consumption
        input_data['Previous_Week_Same_Day'] = prev_week_consumption
        input_data['Festival'] = festival
        input_data['Weather_Encoded'] = weather_mapping[weather]
        
        # Set the correct day to 1
        day_col = f'Day_{day}'
        if day_col in input_data:
            input_data[day_col] = 1
        else:
            logging.warning(f"Day column {day_col} not found in features")
        
        # Create dataframe with correct feature order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[feature_names]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = round(prediction)
        
        # Calculate confidence range (based on MAE of 24)
        lower_bound = max(0, int(prediction - 24))
        upper_bound = int(prediction + 24)
        
        # Log the prediction
        logging.info(f"Prediction made: {prediction} meals")
        
        return render_template('index.html',
                             prediction=prediction,
                             lower_bound=lower_bound,
                             upper_bound=upper_bound,
                             days=days,
                             weather_options=weather_mapping.keys(),
                             form_data=request.form)
    
    except ValueError as e:
        return render_template('index.html',
                             error=f"Invalid input: {str(e)}",
                             days=days,
                             weather_options=weather_mapping.keys())
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return render_template('index.html',
                             error=f"Prediction error: {str(e)}",
                             days=days,
                             weather_options=weather_mapping.keys())

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(feature_names) if feature_names else 0
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)