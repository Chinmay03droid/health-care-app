# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBRegressor

app = Flask(__name__)

# Debug: Print when starting the app
print("Starting Flask application...")

# Load the trained model with error handling
try:
    model_info = pickle.load(open('xgboost_model.pkl', 'rb'))
    model = model_info['model']
    expected_feature_names = model_info['feature_names']
    print("Model loaded successfully!")
    print(f"Expected feature names: {expected_feature_names}")
except Exception as e:
    print(f"\n!!! ERROR LOADING MODEL !!!\n{str(e)}\n")
    # Fallback to old format if the new dictionary format fails
    try:
        model = pickle.load(open('xgboost_model.pkl', 'rb'))
        print("Model loaded using fallback method!")
        # Create default feature names based on your training code pattern
        expected_feature_names = ['age', 'bmi', 'children', 'smoker', 'is_male', 
                                 'region_northeast', 'region_northwest', 'region_southwest']
    except Exception as e2:
        print(f"\n!!! FALLBACK LOADING FAILED !!!\n{str(e2)}\n")
        raise

def standardized_preprocess_df(df):
    """Standardized preprocessing for both training and prediction"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert region to lowercase consistently
    df['region'] = df['region'].str.lower()
    
    # One-hot encode region
    df_new = pd.get_dummies(df, prefix='region', columns=['region'])
    
    # Drop the reference category consistently
    if 'region_southeast' in df_new.columns:
        df_new = df_new.drop(columns=['region_southeast'])
    
    # Handle smoker column consistently
    df_new['smoker'] = df_new['smoker'].map({'yes': 1, 'no': 0})
    
    # Create is_male and drop sex consistently
    df_new['is_male'] = (df_new['sex'] == 'male').astype(int)
    df_new = df_new.drop(columns=['sex'])
    
    return df_new

def preprocess_df(df):
    """Ensures input data matches the model's expected features"""
    # First apply the standard preprocessing
    df_new = standardized_preprocess_df(df)
    
    print("\n=== AFTER STANDARD PREPROCESSING ===")
    print("Current columns:", df_new.columns.tolist())
    
    # Ensure all required columns exist in the correct order
    missing_cols = set(expected_feature_names) - set(df_new.columns)
    if missing_cols:
        print(f"Adding missing columns: {missing_cols}")
        for col in missing_cols:
            df_new[col] = 0
            
    # Drop any extra columns not expected by the model
    extra_cols = set(df_new.columns) - set(expected_feature_names)
    if extra_cols:
        print(f"Dropping extra columns: {extra_cols}")
        df_new = df_new.drop(columns=list(extra_cols))
    
    # Return with columns in the right order
    return df_new[expected_feature_names]

@app.route('/')
def home():
    print("Rendering home page...")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug: Print raw form data
        print("\n=== FORM SUBMISSION RECEIVED ===")
        print("Raw form data:", request.form)
        
        # Get data from form with validation
        try:
            age = int(request.form.get('age'))
            sex = request.form.get('sex').lower()  # Lowercase to match preprocessing
            bmi = float(request.form.get('bmi'))
            children = int(request.form.get('children'))
            smoker = request.form.get('smoker').lower()  # Lowercase to match preprocessing
            region = request.form.get('region').lower()  # Lowercase to match preprocessing
            
            print(f"Parsed values - Age: {age}, Sex: {sex}, BMI: {bmi}, "
                  f"Children: {children}, Smoker: {smoker}, Region: {region}")
        except Exception as e:
            raise ValueError(f"Invalid form data: {str(e)}")

        # Create DataFrame with single row
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })
        
        print("\n=== RAW INPUT DATA ===")
        print(input_data)

        # Process data same as training
        input_processed = preprocess_df(input_data)
        
        print("\n=== PROCESSED DATA ===")
        print("Columns:", input_processed.columns.tolist())
        print("Values:", input_processed.values.tolist())

        # Make prediction
        try:
            prediction = model.predict(input_processed)[0]
            print(f"\nPrediction result: ${prediction:.2f}")
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")

        return render_template('index.html', 
                            prediction_text=f'Estimated Healthcare Charges: ${prediction:.2f}')

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"\n!!! ERROR !!!\n{error_msg}\n")
        return render_template('index.html', prediction_text=error_msg)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Set debug=False in production