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
required_features = [
    'dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate',
    'sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit',
    'swin','dwin','smean','dmean','trans_depth'
]

categorical_cols = ['proto', 'service', 'state']
target = "label"

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
label_enc = LabelEncoder()
scaler = StandardScaler()

# 1. AUTO-CONVERT XLSX → CSV
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

# 2. LOAD DATA
# =========================================================
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

# 3. UNIVERSAL PREPROCESSING FUNCTION
def preprocess_dataset(df, is_training=True):
    df = df.copy()

    # Add missing columns
    for col in required_features + [target]:
        if col not in df.columns:
            print(f"[!] Missing column added: {col}")
            df[col] = 0

    # Drop extra columns 
    df = df[required_features + [target]]

    # Convert numeric columns
    for col in required_features:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Encode categorical
    if is_training:
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols].astype(str))
    else:
        df[categorical_cols] = encoder.transform(df[categorical_cols].astype(str))

    # Encode label
    if is_training:
        df[target] = label_enc.fit_transform(df[target].astype(str))
    else:
        df[target] = label_enc.transform(df[target].astype(str))

    return df

# 4. APPLY PREPROCESSING
train_df = preprocess_dataset(train_df, is_training=True)
test_df = preprocess_dataset(test_df, is_training=False)

# 5. TRAIN/VAL SPLIT
X = train_df[required_features]
y = train_df[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 6. TRAIN MODELS
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}
results = {}

for name, model in models.items():
    print(f"[+] Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    trained_models[name] = model
    results[name] = (y_val, y_pred)
    joblib.dump(model, f"{name.replace(' ', '_')}_model.pkl")

# 7. PRINT RESULTS
for name, (y_true, y_pred) in results.items():
    print("\n========================================")
    print(f" RESULTS: {name}")
    print("========================================")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

# 8. PREDICTION FOR NEW DATA
def test_new_data(df, model_name):
    df = preprocess_dataset(df, is_training=False)
    X_new = df[required_features]
    model = trained_models[model_name]

    preds = model.predict(X_new)
    probs = model.predict_proba(X_new) if hasattr(model,"predict_proba") else None
    return preds, probs
