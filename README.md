# Botnet Traffic Detection System — CSIT375 Project

This project implements a machine learning system for detecting botnet traffic using the UNSW-NB15 dataset.  
It trains multiple models, evaluates their performance, and provides a Tkinter-based GUI for real-time prediction and visualisation.

---

## Features

- Machine-learning-based botnet detection  
- Random Forest, Logistic Regression, and K-Nearest Neighbors  
- Tkinter GUI for user interaction  
- Real-time traffic prediction  
- Confusion matrices (text and heatmap)  
- Prediction performance graphs  
- Stratified K-Fold cross-validation  
- Dataset class distribution visualisation  

---

## Repository Contents

| File | Description |
|------|-------------|
| `GUI.py` | Graphical interface and user interaction logic |
| `Trainer1.py` | Data preprocessing, encoding, scaling, training, and evaluation |
| `UNSW_NB15_training-set.xlsx` | Training dataset |
| `UNSW_NB15_testing-set.xlsx` | Testing dataset |

---

## Machine Learning Models

The system trains and evaluates:

- Random Forest (primary model used for GUI prediction)  
- Logistic Regression  
- K-Nearest Neighbors  

Random Forest is selected as the final operational model because of its consistent accuracy with numerical and categorical network features.

---

## Dataset Processing Pipeline

The data pipeline includes:

1. Merging training and testing datasets  
2. Encoding categorical features using a stored encoder  
3. Scaling numerical features with `StandardScaler`  
4. Applying a 70/30 stratified train-test split  
5. Training and evaluating multiple classifiers  
6. **Applying the same encoding and scaling pipeline inside the GUI during live prediction**  
   to ensure the preprocessing is identical to the training phase

---

## GUI Overview

The Tkinter interface provides several key functions.

### Prediction Graphs
Displays actual versus predicted class counts for each model.

### Confusion Matrices
Outputs confusion matrices both as text and heatmaps.

### Live Traffic Prediction
Users manually input the following important features:

- `dur`  
- `proto`  
- `service`  
- `state`  
- `sbytes`  
- `dbytes`  
- `rate`

All other features required by the model are automatically assigned default values.  
The GUI then encodes and scales the full feature set before prediction.

### Class Distribution
Shows the ratio of benign and botnet samples in the dataset.

### K-Fold Evaluation
Runs stratified K-Fold cross-validation on any selected model.

---

## How to Run

Place these files in the same directory:

- `GUI.py`  
- `Trainer1.py`  
- `UNSW_NB15_training-set.xlsx`  
- `UNSW_NB15_testing-set.xlsx`

### Install dependencies

pip install pandas numpy scikit-learn matplotlib seaborn joblib

Start the GUI
python GUI.py

---

## Notes

- Random Forest is the model used for final traffic prediction in the GUI.  
- Missing or unknown inputs are handled using default values.  
- The GUI uses the same encoding and scaling pipeline as the training phase to ensure consistent preprocessing.  
- This project is developed for academic and research purposes under CSIT375.  




