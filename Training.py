# Plotting and visualization
import matplotlib.pyplot as plt

# Data handling
import pandas as pd

# Model saving/loading
import joblib

# Data splitting
from sklearn.model_selection import train_test_split

# Encoding and scaling
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler

# Machine learning models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# =========================================================
# 1. GLOBAL DEFINITIONS
# =========================================================

# Input features used by the models
required_features = [
    'dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate',
    'sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit',
    'swin','dwin','smean','dmean','trans_depth'
]

# Columns that require categorical encoding
categorical_cols = ['proto', 'service', 'state']

# Output label column
target = "label"

# Encoders and scaler
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)  # Handle unseen categories
label_enc = LabelEncoder()   # Encode target labels
scaler = StandardScaler()    # Normalize feature values


# =========================================================
# 2. EXCEL → CSV AUTO-CONVERSION
# =========================================================

# Converts .xlsx file to .csv if needed
def convert_excel_to_csv(path):
    if path.endswith(".xlsx"):                         # Check file type
        csv_path = path.replace(".xlsx", ".csv")       # Create CSV name
        print(f"[+] Converting {path} → {csv_path}")   # Log conversion
        pd.read_excel(path).to_csv(csv_path, index=False)  # Save as CSV
        return csv_path
    return path


# Convert both datasets if needed
train_path = convert_excel_to_csv("UNSW_NB15_training-set.xlsx")
test_path  = convert_excel_to_csv("UNSW_NB15_testing-set.xlsx")


# =========================================================
# 3. LOAD DATASETS
# =========================================================

# Load both CSV files
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

# Clean column names
train_df.columns = train_df.columns.str.strip()
test_df.columns  = test_df.columns.str.strip()

# Combine training and testing files into one dataset
full_df = pd.concat([train_df, test_df], ignore_index=True)


# =========================================================
# 4. PREPROCESSING FUNCTION
# =========================================================
def preprocess(df, is_training=True):
    df = df.copy()    # Work on a copy to protect original data

    # Add missing required columns (if any)
    for col in required_features + [target]:
        if col not in df.columns:
            print(f"[!] Missing column added: {col}")
            df[col] = 0

    # Keep only required columns
    df = df[required_features + [target]]

    # Convert numerical columns to numeric and fill missing values
    for col in required_features:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Encode categorical columns
    if is_training:
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols].astype(str))
    else:
        df[categorical_cols] = encoder.transform(df[categorical_cols].astype(str))

    # Encode labels
    if is_training:
        df[target] = label_enc.fit_transform(df[target].astype(str))
    else:
        df[target] = label_enc.transform(df[target].astype(str))

    return df


# =========================================================
# 5. APPLY PREPROCESSING
# =========================================================

# Preprocess the full dataset
full_df = preprocess(full_df, is_training=True)

# Plots class distribution for imbalance checking
def plot_class_distribution(df):
    """
    df: pandas DataFrame containing the 'label' column
    Plots a bar graph showing the number of 0s (benign) vs 1s (botnet)
    """
    if 'label' not in df.columns:
        print("[!] No 'label' column in dataframe.")
        return

    counts = df['label'].value_counts().sort_index()
    plt.figure(figsize=(6,4))
    plt.bar(['Benign (0)', 'Botnet (1)'], counts, color=['green', 'red'])
    plt.title("Class Distribution")
    plt.ylabel("Number of Samples")
    plt.show()


# Cross-validation tools
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Perform K-Fold Cross Validation
def kfold_evaluation(model, X, y, k=7):
    """
    Perform K-Fold cross-validation on a given model.
    """
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)  # Stratified for class balance
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    mean_acc = scores.mean()
    std_acc = scores.std()

    print(f"[+] K-Fold Evaluation (K={k})")
    print("Scores per fold:", scores)
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Standard Deviation: {std_acc:.4f}\n")

    return scores, mean_acc, std_acc


# =========================================================
# 6. TRAIN-TEST SPLIT (70/30)
# =========================================================

# Separate features and target
X = full_df[required_features]
y = full_df[target]

# Stratified split to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Fit scaler on training data and transform both sets
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# =========================================================
# 7. TRAIN MODELS
# =========================================================

# Models with class balance handling
models = {
    "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced"),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}

# Train, store and save each model
for name, model in models.items():
    print(f"\n[+] Training: {name}")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    joblib.dump(model, f"{name.replace(' ', '_')}.pkl")   # Save model to file


# =========================================================
# 8. EVALUATE MODELS
# =========================================================

results = {}  # Dictionary to store true and predicted values

# Evaluate each trained model
for name, model in trained_models.items():
    print("\n============================================")
    print(f" RESULTS FOR: {name}")
    print("============================================")

    preds = model.predict(X_test_scaled)   # Make predictions

    # Store predictions for later use (e.g. GUI)
    results[name] = (y_test, preds)

    # Calculate misclassification rate
    misclassified = (preds != y_test).sum()
    total = len(y_test)
    mis_rate = misclassified / total

    print(f"Misclassified Samples: {misclassified} / {total}")
    print(f"Misclassification Rate: {mis_rate:.4f}")

    # Display standard evaluation metrics
    print("\nAccuracy:", accuracy_score(y_test, preds))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))



