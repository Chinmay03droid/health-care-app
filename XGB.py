import os
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load your dataset with the correct path
insurance = pd.read_csv('C:/Users/Asus/insurance.csv')

# Clean the charges column by removing $ signs and commas
if isinstance(insurance['charges'].iloc[0], str):
    insurance['charges'] = insurance['charges'].astype(str)
    insurance['charges'] = insurance['charges'].str.replace('$', '', regex=False)
    insurance['charges'] = insurance['charges'].str.replace(',', '', regex=False)

# Convert 'charges' to numeric, forcing errors to NaN
insurance['charges'] = pd.to_numeric(insurance['charges'], errors='coerce')

# Fill NaN values if there are any
insurance['charges'] = insurance['charges'].fillna(insurance['charges'].mean())

# Standardized preprocessing function - SAME as in Flask app
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

# Process the data
processed_df = standardized_preprocess_df(insurance)

# Define features and target
X = processed_df.drop(columns=['charges'])
y = processed_df['charges']

# Save the feature names for prediction
feature_names = X.columns.tolist()
print("Feature names:", feature_names)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Create a dictionary containing both the model and feature names
model_info = {
    'model': model,
    'feature_names': feature_names
}

# Save the model to the specified path
save_path = 'C:/Users/Asus/Downloads/Projects/xgboost_model.pkl'
with open(save_path, 'wb') as model_file:
    pickle.dump(model_info, model_file)

print(f"Model saved successfully to {save_path}!")
print(f"Feature names: {feature_names}")

# Optional: Test predictions on validation data
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f"Model R² score: {r2:.4f}")