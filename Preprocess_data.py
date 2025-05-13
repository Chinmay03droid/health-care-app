import pandas as pd
import pickle

# Load the model with the new dictionary format
try:
    with open('C:/Users/Asus/Downloads/Projects/xgboost_model.pkl', 'rb') as file:
        model_info = pickle.load(file)
    
    # Extract model and feature names
    if isinstance(model_info, dict) and 'model' in model_info:
        model = model_info['model']
        expected_columns = model_info['feature_names']
        print("Loaded model from dictionary format")
    else:
        # Fallback for old format
        model = model_info
        expected_columns = model.get_booster().feature_names
        print("Loaded model directly")
    
    print(f"Expected columns: {expected_columns}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Standardized preprocessing function
def standardized_preprocess_df(df):
    df = df.copy()
    df['region'] = df['region'].str.lower()
    df_new = pd.get_dummies(df, prefix='region', columns=['region'])
    
    if 'region_southeast' in df_new.columns:
        df_new = df_new.drop(columns=['region_southeast'])
    
    df_new['smoker'] = df_new['smoker'].map({'yes': 1, 'no': 0})
    df_new['is_male'] = (df_new['sex'] == 'male').astype(int)
    df_new = df_new.drop(columns=['sex'])
    
    return df_new

# Function to ensure features match expected columns
def preprocess_df(df):
    df_new = standardized_preprocess_df(df)
    
    # Ensure all expected columns exist
    for col in expected_columns:
        if col not in df_new.columns:
            print(f"Adding missing column: {col}")
            df_new[col] = 0
    
    # Remove extra columns
    extra_cols = set(df_new.columns) - set(expected_columns)
    if extra_cols:
        print(f"Removing extra columns: {extra_cols}")
        df_new = df_new.drop(columns=list(extra_cols))
    
    # Return with columns in correct order
    return df_new[expected_columns]

# Test data for prediction
test_data = pd.DataFrame({
    'age': [25, 40, 60],
    'bmi': [22, 28, 30],
    'children': [0, 2, 3],
    'smoker': ['yes', 'no', 'yes'],
    'sex': ['female', 'male', 'female'],
    'region': ['northwest', 'northeast', 'southwest']
})

# Preprocess and make predictions
test_data_processed = preprocess_df(test_data)
print("\nProcessed test data:")
print(test_data_processed.head())

# Make predictions
predictions = model.predict(test_data_processed)

# Print the predictions
for i, pred in enumerate(predictions):
    print(f"Prediction for entry {i+1}: ${pred:.2f}")





