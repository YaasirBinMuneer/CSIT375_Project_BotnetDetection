# Import necessary libraries for GUI, data processing, and visualization
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import custom training module that contains the ML models and data processing
import Training as trainer  

# Class to redirect stdout (print statements) to a text widget in the GUI terminal
class RedirectText:
    def __init__(self, text_ctrl):
        self.output = text_ctrl  # Text widget to write logs into

    def write(self, string):
        self.output.configure(state='normal') # Enable editing to insert text
        self.output.insert(tk.END, string) # Append the string to the end
        self.output.see(tk.END)  # Scroll to the end to show latest output
        self.output.configure(state='disabled') # Disable editing again

    def flush(self):
        pass # Required for stdout redirection but not actually used

# GUI Setup: Create the main window and configure its appearance
root = tk.Tk() # Create the root window (main GUI window)
root.title("Botnet Traffic Detection GUI") # Set window title
root.geometry("950x500") # Set window size
root.configure(bg="#2e2e2e")  # Dark background colour

# Define font styles and colors for consistency
FONT_TITLE = ("Arial", 16, "bold")
FONT_LABEL = ("Arial", 11)
BG_COLOR = "#2e2e2e"  # Dark background colour
FG_COLOR = "#f5f5f5"  # Light text color
ENTRY_BG = "#3e3e3e"  # Darker background for entry fields
ENTRY_FG = "#f5f5f5"  # Light text for entries

# Create main frame to hold all GUI elements
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Create frame for the terminal/log window on the left
terminal_frame = tk.Frame(main_frame, bg=BG_COLOR)
terminal_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create frame for buttons on the right
button_frame = tk.Frame(main_frame, bg=BG_COLOR)
button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# Create the terminal/log text window widget and redirect stdout to it
log_text = tk.Text(terminal_frame, state='disabled', width=60, bg="#1e1e1e", fg="#f5f5f5", font=("Consolas", 10))
log_text.pack(fill=tk.BOTH, expand=True)
sys.stdout = RedirectText(log_text)

# Add a title label to the button frame
title_label = tk.Label(button_frame, text="Botnet Traffic Detection", font=FONT_TITLE, bg=BG_COLOR, fg=FG_COLOR)
title_label.pack(pady=10)

# Function to display model performance graphs
# Button functions
def show_graphs():
    print("[+] Displaying model performance graphs...") # Log to terminal
    plt.figure(figsize=(10, 5)) # Create a figure for plotting
    for name, (y_true, y_pred) in trainer.results.items(): # Iterate over model results
        # Count number of predicted and actual labels 
        actual_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        # Plot actual counts with circle markers
        plt.plot(actual_counts.index, actual_counts.values, marker='o', label=f"{name} Actual")
        # Plot predicted counts with x markers and dashed lines
        plt.plot(pred_counts.index, pred_counts.values, marker='x', linestyle='--', label=f"{name} Predicted")

    # Set plot labels and title
    plt.xlabel("Class (0=Benign, 1=Botnet)")
    plt.ylabel("Count")
    plt.title("Predicted vs Actual Counts for Each Model")
    plt.legend() # Show legend
    plt.grid(True) # Add grid
    plt.tight_layout() # Adjust layout
    plt.show() # Display the plot

# Function to show confusion matrices and classification reports to terminal
def show_confusion_matrix():
    print("\n[+] Confusion Matrices & Classification Reports:") # Log to terminal
    for name, (y_true, y_pred) in trainer.results.items(): # Iterate over model results
        print(f"\n--- {name} ---") # Print model name
        print("Confusion Matrix:")
        print(trainer.confusion_matrix(y_true, y_pred))  # Print confusion matrix
        print("Classification Report:")
        print(trainer.classification_report(y_true, y_pred)) # Print classification report

    print("[+] Displaying confusion matrix graphs...") # Log to terminal
    # Create a subplot for each model's confusion matrix
    plt.figure(figsize=(12, 4))  # Create a smaller figure for multiple heatmaps
    for i, (name, (y_true, y_pred)) in enumerate(trainer.results.items(), 1):  # Iterate with index
        cm = trainer.confusion_matrix(y_true, y_pred)  # Get confusion matrix
        plt.subplot(1, 3, i) # 1 row, 3 columns, current subplot position for each model
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False) # Plot heatmap
        plt.title(f"{name}", fontsize=12) # Set title
        plt.xlabel("Predicted")
        plt.ylabel("Actual") 
    plt.tight_layout() # Adjust layout
    plt.show() # Display the plots
    
# Function to open a form for predicting traffic type
def open_prediction_form():
    # Create a new top-level window for the prediction form
    form_window = tk.Toplevel(root)
    form_window.title("Predict Traffic")
    form_window.geometry("400x700")
    form_window.configure(bg=BG_COLOR)

    # Dictionary to store entry widgets for each feature
    entries = {}
    
    # Create a scrollable canvas and frame for the form inputs (in case of many features)
    canvas = tk.Canvas(form_window, bg=BG_COLOR)
    scrollbar = tk.Scrollbar(form_window, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg=BG_COLOR)

    # Bind the scroll region to the frame's size
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create labels and entry fields for each feature
    for feature in trainer.features:
        # Label for the feature
        lbl = tk.Label(scroll_frame, text=feature, bg=BG_COLOR, fg=FG_COLOR, font=FONT_LABEL)
        lbl.pack(anchor='w', padx=5, pady=2) # Pack label
        # Entry field for user input
        ent = tk.Entry(scroll_frame, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
        ent.pack(fill=tk.X, padx=5, pady=2) # Pack entry
        entries[feature] = ent # Store entry in dictionary

    # Nested function to handle prediction using trained model
    def predict_traffic():
        input_values = [] # List to hold input values
        # Process each feature value from the form
        for feature in trainer.features:
            val = entries[feature].get() # Get value from entry
            # Handle categorical vs numerical features differently
            if feature in trainer.categorical_cols: # If categorical, keep as string
                input_values.append(val)
            else:  # If numerical, convert to float
                try:
                    input_values.append(float(val))
                except: # Default to 0.0 if conversion fails
                    input_values.append(0.0) # Default to 0.0 if conversion fails

        # Create DataFrame from input data
        X_new = pd.DataFrame([input_values], columns=trainer.features)
        # Encode categorical variables using the same encoder from training
        X_new[trainer.categorical_cols] = trainer.encoder.transform(X_new[trainer.categorical_cols].astype(str))

        # Use Random Forest model for prediction
        model = trainer.trained_models["Random Forest"]
        pred_label = model.predict(X_new) # Predict label
        pred_text = trainer.label_enc.inverse_transform(pred_label)[0] # Convert to text

        # Show prediction in a message box and log to terminal
        messagebox.showinfo("Prediction", f"Predicted Traffic Type: {pred_text}")
        print(f"[+] Predicted Traffic Type: {pred_text}")

    # Add submit button to the prediction form
    submit_btn = ttk.Button(scroll_frame, text="Submit Prediction", command=predict_traffic)
    submit_btn.pack(pady=10)

# Configure button style
style = ttk.Style()
style.configure("TButton", font=("Arial", 11, "bold"), padding=5)

# Create buttons for each function of graph display
btn_graphs = ttk.Button(button_frame, text="Show Graphs", command=show_graphs)
btn_graphs.pack(pady=5)

# Create buttons for confusion matrix 
btn_cm = ttk.Button(button_frame, text="Show Confusion Matrix", command=show_confusion_matrix)
btn_cm.pack(pady=5)

# Create buttons for prediction form
btn_predict = ttk.Button(button_frame, text="Predict Traffic", command=open_prediction_form)
btn_predict.pack(pady=5)

# Launch GUI and display initial message
print("[+] Dark GUI Loaded. Terminal on the left. Buttons on the right. Click 'Predict Traffic' to open the form.")
root.mainloop() # Start the Tkinter event loop

