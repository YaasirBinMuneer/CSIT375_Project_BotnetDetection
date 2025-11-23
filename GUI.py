# Import necessary libraries for GUI, data processing, and visualization
import tkinter as tk
from tkinter import ttk, messagebox
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import custom training module that contains the ML models and data processing
import Trainer1 as trainer  # your trainer.py

# =============================
# Redirect stdout to terminal
# =============================
# Class to redirect print statements to the GUI text widget (terminal/log window)
class RedirectText:
    def __init__(self, text_ctrl):
        # Text widget to write logs into
        self.output = text_ctrl

    def write(self, string):
        # Enable editing to insert text
        self.output.configure(state='normal')
        # Append the string to the end
        self.output.insert(tk.END, string)
        # Scroll to the end to show the latest output
        self.output.see(tk.END)
        # Disable editing again
        self.output.configure(state='disabled')

    def flush(self):
        # Required for stdout redirection but not actually used
        pass

# =============================
# GUI Setup
# =============================
# Create the root window (main GUI window)
root = tk.Tk()
# Set window title
root.title("Botnet Traffic Detection GUI")
# Set window size
root.geometry("950x500")
# Dark background colour
root.configure(bg="#2e2e2e")

# Fonts & colors for consistency
FONT_TITLE = ("Arial", 16, "bold")
FONT_LABEL = ("Arial", 11)
BG_COLOR = "#2e2e2e"         # Dark background colour
FG_COLOR = "#f5f5f5"         # Light text color
ENTRY_BG = "#3e3e3e"         # Darker background for entry fields
ENTRY_FG = "#f5f5f5"         # Light text for entries

# =============================
# Layout Frames
# =============================
# Create main frame to hold all GUI elements
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Create frame for the terminal/log window on the left
terminal_frame = tk.Frame(main_frame, bg=BG_COLOR)
terminal_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create frame for buttons on the right
button_frame = tk.Frame(main_frame, bg=BG_COLOR)
button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# =============================
# Terminal / Log Window
# =============================
# Create the terminal/log text window widget and redirect stdout to it
log_text = tk.Text(
    terminal_frame,
    state='disabled',
    width=60,
    bg="#1e1e1e",
    fg="#f5f5f5",
    font=("Consolas", 10)
)
log_text.pack(fill=tk.BOTH, expand=True)
sys.stdout = RedirectText(log_text)

# =============================
# Title
# =============================
# Add a title label to the button frame
title_label = tk.Label(
    button_frame,
    text="Botnet Traffic Detection",
    font=FONT_TITLE,
    bg=BG_COLOR,
    fg=FG_COLOR
)
title_label.pack(pady=10)

# =============================
# Button Functions
# =============================
# Function to display model performance graphs
def show_graphs():
    # Log to terminal
    print("[+] Displaying model performance graphs...")

    # Create a figure for plotting predicted vs actual values
    plt.figure(figsize=(10, 5))

    # Iterate over model results
    for name, (y_true, y_pred) in trainer.results.items():
        # Count number of predicted and actual labels
        actual_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()

        # Plot actual counts with circle markers
        plt.plot(
            actual_counts.index,
            actual_counts.values,
            marker='o',
            label=f"{name} Actual"
        )

        # Plot predicted counts with x markers and dashed lines
        plt.plot(
            pred_counts.index,
            pred_counts.values,
            marker='x',
            linestyle='--',
            label=f"{name} Predicted"
        )

    # Set plot labels and title
    plt.xlabel("Class (0=Benign, 1=Botnet)")
    plt.ylabel("Count")
    plt.title("Predicted vs Actual Counts for Each Model")

    # Show legend
    plt.legend()

    # Add grid
    plt.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    plt.show()


# Function to show confusion matrices and classification reports to terminal and as graphs
def show_confusion_matrix():
    # Log to terminal
    print("\n[+] Confusion Matrices & Classification Reports:")

    # Iterate over model results
    for name, (y_true, y_pred) in trainer.results.items():
        # Print model name
        print(f"\n--- {name} ---")

        # Print confusion matrix
        print("Confusion Matrix:")
        print(trainer.confusion_matrix(y_true, y_pred))

        # Print classification report
        print("Classification Report:")
        print(trainer.classification_report(y_true, y_pred))

    # Log to terminal
    print("[+] Displaying confusion matrix graphs...")

    # Create a smaller figure for multiple heatmaps
    plt.figure(figsize=(12, 4))

    # Iterate with index for subplot positioning
    for i, (name, (y_true, y_pred)) in enumerate(trainer.results.items(), 1):
        # Get confusion matrix
        cm = trainer.confusion_matrix(y_true, y_pred)

        # 1 row, 3 columns, current subplot position for each model
        plt.subplot(1, 3, i)

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)

        # Set title and labels
        plt.title(f"{name}", fontsize=12)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

    # Adjust layout
    plt.tight_layout()

    # Display the plots
    plt.show()


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

    # Important features only (subset of total features)
    important_features = ['dur', 'proto', 'service', 'state', 'sbytes', 'dbytes', 'rate']

    # Create labels and entry fields for each feature
    for feature in important_features:
        # Label for the feature
        lbl = tk.Label(
            scroll_frame,
            text=feature,
            bg=BG_COLOR,
            fg=FG_COLOR,
            font=FONT_LABEL
        )
        lbl.pack(anchor='w', padx=5, pady=2)

        # Entry field for user input
        ent = tk.Entry(
            scroll_frame,
            bg=ENTRY_BG,
            fg=ENTRY_FG,
            insertbackground=FG_COLOR
        )
        ent.pack(fill=tk.X, padx=5, pady=2)

        # Store entry in dictionary
        entries[feature] = ent

    # Nested function to handle prediction using trained model
    def predict_traffic():
        # List to hold input values
        input_values = {}

        # 1️⃣ Collect user inputs for important features
        for feature in important_features:
            # Get value from entry
            val = entries[feature].get()

            # Handle categorical vs numerical features differently
            if feature in trainer.categorical_cols:
                # If categorical, keep as string
                input_values[feature] = val if val else 'unknown'
            else:
                # If numerical, convert to float
                try:
                    input_values[feature] = float(val)
                except:
                    # Default to 0.0 if conversion fails
                    input_values[feature] = 0.0

        # 2️⃣ Fill in remaining features with default values
        for feature in trainer.required_features:
            if feature not in input_values:
                if feature in trainer.categorical_cols:
                    input_values[feature] = 'unknown'
                else:
                    input_values[feature] = 0.0

        # 3️⃣ Create DataFrame from input data in the correct column order
        X_new = pd.DataFrame([input_values], columns=trainer.required_features)

        # 4️⃣ Encode categorical variables using the same encoder from training
        X_new[trainer.categorical_cols] = trainer.encoder.transform(
            X_new[trainer.categorical_cols].astype(str)
        )

        # 5️⃣ Use Random Forest model for prediction
        model = trainer.trained_models["Random Forest"]
        pred_label = model.predict(X_new)

        # 6️⃣ Convert numeric label to text ("Benign" or "Botnet")
        pred_text = trainer.label_enc.inverse_transform(pred_label)[0]

        # 7️⃣ Show prediction in a message box and log to terminal
        messagebox.showinfo("Prediction", f"Predicted Traffic Type: {pred_text}")
        print(f"[+] Predicted Traffic Type: {pred_text}")

    # Add submit button to the prediction form
    submit_btn = ttk.Button(scroll_frame, text="Submit Prediction", command=predict_traffic)
    submit_btn.pack(pady=10)


# Function to show class distribution in a popup window
def show_class_dist_popup():
    # Create popup window
    popup = tk.Toplevel(root)
    popup.title("Class Distribution")
    popup.geometry("400x300")

    # Create figure for class distribution
    fig, ax = plt.subplots(figsize=(4, 3))

    # Count labels in training data
    counts = trainer.train_df['label'].value_counts().sort_index()

    # Plot class distribution as bar chart
    ax.bar(['Benign (0)', 'Botnet (1)'], counts, color=['green', 'red'])

    # Set titles and labels
    ax.set_ylabel("Number of Samples")
    ax.set_title("Class Distribution")

    # Embed matplotlib figure into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=popup)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)


# Function to open K-Fold evaluation popup
def open_kfold_popup():
    # Create popup window
    popup = tk.Toplevel(root)
    popup.title("K-Fold Evaluation")
    popup.geometry("350x200")
    popup.configure(bg=BG_COLOR)

    # ---------- Model Selection ----------
    # Dropdown to select which trained model to evaluate
    tk.Label(
        popup,
        text="Select Model:",
        bg=BG_COLOR,
        fg=FG_COLOR,
        font=FONT_LABEL
    ).pack(pady=5)

    model_var = tk.StringVar()
    model_var.set("Random Forest")  # default selection

    model_options = list(trainer.trained_models.keys())
    model_menu = ttk.Combobox(
        popup,
        textvariable=model_var,
        values=model_options,
        state="readonly"
    )
    model_menu.pack(pady=5)

    # ---------- K Input ----------
    # Input field for number of folds (K)
    tk.Label(
        popup,
        text="Enter K (Number of Folds):",
        bg=BG_COLOR,
        fg=FG_COLOR,
        font=FONT_LABEL
    ).pack(pady=5)

    k_entry = tk.Entry(popup, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
    k_entry.insert(0, "7")  # default value
    k_entry.pack(pady=5)

    # ---------- Evaluate Button ----------
    # Function to run K-Fold evaluation when button is clicked
    def run_kfold():
        model_name = model_var.get()

        # Validate K input
        try:
            k = int(k_entry.get())
            if k < 2:
                raise ValueError
        except:
            messagebox.showerror("Invalid Input", "Please enter a valid integer K >= 2")
            return

        # Get selected model
        model = trainer.trained_models[model_name]

        # Log to terminal
        print(f"\n[+] Running K-Fold Evaluation for {model_name} with K={k}...")

        # Perform K-Fold evaluation
        scores, mean_acc, std_acc = trainer.kfold_evaluation(
            model,
            trainer.X,
            trainer.y,
            k
        )

        # Done message
        print(f"[+] Done!\n")

    eval_btn = ttk.Button(popup, text="Evaluate", command=run_kfold)
    eval_btn.pack(pady=10)


# =============================
# Buttons
# =============================
# Configure button style
style = ttk.Style()
style.configure("TButton", font=("Arial", 11, "bold"), padding=5)

# Create buttons for each function of graph display
btn_graphs = ttk.Button(
    button_frame,
    text="Prediction \nGraphs",
    command=show_graphs
)
btn_graphs.pack(pady=5)

# Create button for confusion matrices
btn_cm = ttk.Button(
    button_frame,
    text="Confusion Matrices",
    command=show_confusion_matrix
)
btn_cm.pack(pady=5)

# Create button for prediction form
btn_predict = ttk.Button(
    button_frame,
    text="Predict Traffic",
    command=open_prediction_form
)
btn_predict.pack(pady=5)

# Create button for dataset class distribution
btn_class_dist = tk.Button(
    button_frame,
    text="Dataset\nClass Distribution",
    command=show_class_dist_popup
)
btn_class_dist.pack(pady=5)

# Create button for K-Fold evaluation
btn_kfold = ttk.Button(
    button_frame,
    text="K-Fold Evaluation",
    command=open_kfold_popup
)
btn_kfold.pack(pady=5)


# =============================
# Launch GUI
# =============================
# Initial message in terminal
print("[+] Dark GUI Loaded. Terminal on the left. Buttons on the right. Click 'Predict Traffic' to open the form.")

# Start the Tkinter event loop
root.mainloop()
