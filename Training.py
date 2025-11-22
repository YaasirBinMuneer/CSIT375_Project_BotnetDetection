# Import necessary libraries for data manipulation, file handling, machine learning, and model persistence
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 0. DEFINE GLOBALS
# Define the list of required features (columns) for the model input
required_features = [
    'dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate',
    'sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit',
    'swin','dwin','smean','dmean','trans_depth'
]

# Specify which features are categorical
categorical_cols = ['proto', 'service', 'state']
# Define the target column (the label to predict)
target = "label"

# Initialize encoders and scaler for preprocessing
# OrdinalEncoder for categorical features, handling unknown values
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
# LabelEncoder for the target variable
label_enc = LabelEncoder()
# StandardScaler for numerical features (initialized but not used in this code)
scaler = StandardScaler()

# 1. AUTO-CONVERT XLSX → CSV
# Function to automatically convert Excel (.xlsx) files to CSV format if needed
def convert_excel_to_csv(excel_path):
    if excel_path.endswith(".xlsx"): # Check if the file is an Excel file
        csv_path = excel_path.replace(".xlsx", ".csv") # Generate the corresponding CSV path by replacing the extension
        print(f"[+] Converting {excel_path} → {csv_path}") # Log the conversion
        df = pd.read_excel(excel_path) # Read the Excel file into a DataFrame
        df.to_csv(csv_path, index=False) # Save the DataFrame as a CSV file without the index
        return csv_path # Return the CSV path
    return excel_path # If it's not an Excel file, return the original path

# Convert the training and testing Excel files to CSV if they exist
train_path = convert_excel_to_csv("UNSW_NB15_training-set.xlsx")
test_path = convert_excel_to_csv("UNSW_NB15_testing-set.xlsx")

# 2. LOAD DATA
# Load the training and testing datasets from the CSV files
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Strip any leading/trailing whitespace from column names to avoid issues
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# 3. UNIVERSAL PREPROCESSING FUNCTION
# Define a function to preprocess the dataset, handling both training and testing data
def preprocess_dataset(df, is_training=True):
    df = df.copy() # Create a copy of the DataFrame to avoid modifying the original

    # Add any missing required columns or target column, filling with 0 if absent
    for col in required_features + [target]:
        if col not in df.columns:
            print(f"[!] Missing column added: {col}") # Log missing columns
            df[col] = 0 # Add the column with default value 0

    # Drop any extra columns not in the required features or target
    df = df[required_features + [target]]

    # Convert numerical columns to numeric types, filling NaNs with 0
    for col in required_features:
        if col not in categorical_cols: # Skip categorical columns
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Encode categorical
    if is_training:
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols].astype(str))
    else:
        df[categorical_cols] = encoder.transform(df[categorical_cols].astype(str))

    # Encode target labels (benign/malicious)
    if is_training:
        # Fit and transform during training
        df[target] = label_enc.fit_transform(df[target].astype(str))
    else:
        # Only transform during testing (using fitted label encoder)
        df[target] = label_enc.transform(df[target].astype(str))

    # Return the preprocessed DataFrame
    return df

# 4. APPLY PREPROCESSING
# Apply preprocessing to the training and testing datasets
train_df = preprocess_dataset(train_df, is_training=True) # Fit encoders on training data
test_df = preprocess_dataset(test_df, is_training=False) # Use fitted encoders on testing data

# 5. TRAIN/VAL SPLIT
# Separate features (X) and target (y) from the preprocessed training data
X = train_df[required_features]
y = train_df[target]

# Split the training data into training and validation sets (75% train, 25% val)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42 # Seed for reproducible splits
)

# 6. TRAIN MACHINE LEARNING MODELS
# Define a dictionary of models to train
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42), # Random Forest with 100 trees
    "Logistic Regression": LogisticRegression(max_iter=1000), # Logistic Regression with increased iterations for convergance
    "KNN": KNeighborsClassifier(n_neighbors=5) # KNN with 5 neighbors
}

# Dictionaries to store trained models and their results
trained_models = {}
results = {}

# Train each model, make predictions on validation set, and save the model
for name, model in models.items():
    print(f"[+] Training {name}...") # Log the training process
    model.fit(X_train, y_train) # Fit the model on training data
    y_pred = model.predict(X_val)  # Predict on validation data
    trained_models[name] = model # Store the trained model
    results[name] = (y_val, y_pred) # Store true and predicted labels for evaluation
    joblib.dump(model, f"{name.replace(' ', '_')}_model.pkl") # Save the model to a file using joblib

# 7. PRINT AND EVALUATE MODEL RESULTS
# Print evaluation metrics for each trained model
for name, (y_true, y_pred) in results.items():
    print("\n========================================")
    print(f" RESULTS: {name}") # Header for the model's results
    print("========================================")
    print("Accuracy:", accuracy_score(y_true, y_pred)) # Print accuracy score
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred)) # Print confusion matrix
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred)) # Print detailed classification report

# 8. PREDICTION FOR NEW DATA
# Function to preprocess new data and make predictions using a specified model
def test_new_data(df, model_name):
    # Preprocess the new data (not training, so use fitted encoders)
    df = preprocess_dataset(df, is_training=False)
    X_new = df[required_features] # Extract features
    model = trained_models[model_name] # Get the specified trained model

    # Make predictions
    preds = model.predict(X_new)
    # Get prediction probabilities if the model supports it
    probs = model.predict_proba(X_new) if hasattr(model,"predict_proba") else None
    # Return predictions and probabilities
    return preds, probs

