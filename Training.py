import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =============================
# 1. AUTO-CONVERT XLSX → CSV
# =============================

def convert_excel_to_csv(excel_path):
    if excel_path.endswith(".xlsx"):
        csv_path = excel_path.replace(".xlsx", ".csv")
        print(f"[+] Converting {excel_path} → {csv_path}")
        df = pd.read_excel(excel_path)
        df.to_csv(csv_path, index=False)
        return csv_path
    return excel_path

train_path = convert_excel_to_csv("UNSW_NB15_training-set.xlsx")
test_path = convert_excel_to_csv("UNSW_NB15_testing-set.xlsx")

# =============================
# 2. LOAD DATA
# =============================
print("[+] Loading datasets...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Clean column names
train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# =============================
# 3. SELECT FEATURES
# =============================
features = [
    'dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate',
    'sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit',
    'swin','dwin','smean','dmean','trans_depth'
]

target = 'label'

# =============================
# 4. ENCODE CATEGORICAL VALUES SAFELY
# =============================
categorical_cols = ['proto', 'service', 'state']

# OrdinalEncoder handles unseen categories safely
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_df[categorical_cols] = encoder.fit_transform(train_df[categorical_cols].astype(str))
test_df[categorical_cols] = encoder.transform(test_df[categorical_cols].astype(str))

# Encode label
label_enc = LabelEncoder()
train_df[target] = label_enc.fit_transform(train_df[target].astype(str))
test_df[target] = label_enc.transform(test_df[target].astype(str))

# =============================
# 5. SPLIT TRAIN/VALIDATION
# =============================
X = train_df[features]
y = train_df[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# =============================
# 6. TRAIN MODELS
# =============================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}
results = {}

print("\n==============================")
print(" TRAINING MODELS ")
print("==============================")

for name, model in models.items():
    print(f"[+] Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    trained_models[name] = model
    results[name] = (y_val, y_pred)
    # Save model
    joblib.dump(model, f"{name.replace(' ', '_')}_model.pkl")

print("\n[+] All models trained and saved as .pkl files!")

# =============================
# 7. PRINT RESULTS
# =============================
for name, (y_true, y_pred) in results.items():
    print("\n========================================")
    print(f" RESULTS: {name}")
    print("========================================")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# =============================
# 8. TESTING FUNCTION
# =============================
def test_new_data(df, features, model_name):
    """
    df: new DataFrame with same columns
    features: list of feature columns
    model_name: one of the trained model keys
    """
    if model_name not in trained_models:
        print(f"Model {model_name} not found!")
        return

    # Encode categorical columns
    df[categorical_cols] = encoder.transform(df[categorical_cols].astype(str))

    X_new = df[features]
    model = trained_models[model_name]
    preds = model.predict(X_new)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_new)

    return preds, probs

# Example usage:
# preds, probs = test_new_data(test_df, features, "Random Forest")
