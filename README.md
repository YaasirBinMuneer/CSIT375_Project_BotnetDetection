# CSIT375_Project_BotnetDetection
# Botnet Traffic Detection System (GUI + Machine Learning)

## Overview
This project implements a **botnet traffic detection system** using machine learning and a **Tkinter-based GUI**.  
The system trains multiple models on the **UNSW-NB15 dataset** and allows users to:

- Visualize model performance
- View confusion matrices
- Perform live traffic prediction
- Run K-Fold cross-validation
- View class distribution

---

## Files in the Repository
- `GUI.py` – Graphical user interface and user interaction
- `Trainer1.py` – Data processing, model training, evaluation
- `UNSW_NB15_training-set.xlsx` – Training dataset
- `UNSW_NB15_testing-set.xlsx` – Testing dataset

---

## Models Used
- **Random Forest** *(main model used for live prediction)*
- Logistic Regression
- K-Nearest Neighbors (KNN)

---

## Dataset Handling
- Training and testing files are combined
- Categorical features are encoded using an encoder
- Numerical features are scaled using `StandardScaler`
- Data is split into **70% training / 30% testing** using **stratified sampling**

---

## GUI Features
- **Prediction Graphs** – Actual vs predicted values for each model
- **Confusion Matrices** – Displayed as text and heatmaps
- **Predict Traffic** – Input form for live prediction
- **Dataset Class Distribution** – Bar chart of labels
- **K-Fold Evaluation** – User-selected model and K value

---

## Features Used for Live Prediction
The GUI requires only these main inputs:
dur, proto, service, state, sbytes, dbytes, rate

All remaining features are automatically filled with default values by the system.

---

## How to Run

1. Ensure all files are in the same directory:
   - `GUI.py`
   - `Trainer1.py`
   - `UNSW_NB15_training-set.xlsx`
   - `UNSW_NB15_testing-set.xlsx`

2. Install required libraries:
pip install pandas numpy scikit-learn matplotlib seaborn joblib

3. Run the application:
python GUI.py

---

## Notes
- Random Forest is used for final traffic prediction in the GUI
- K-Fold uses **Stratified K-Fold** cross-validation
- Missing or unknown values are handled using default values
- Project is designed for academic and research purposes
