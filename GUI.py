import tkinter as tk
from tkinter import ttk, messagebox
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import Training as trainer  # your trainer.py

class RedirectText:
    def __init__(self, text_ctrl):
        self.output = text_ctrl

    def write(self, string):
        self.output.configure(state='normal')
        self.output.insert(tk.END, string)
        self.output.see(tk.END)
        self.output.configure(state='disabled')

    def flush(self):
        pass


root = tk.Tk()
root.title("Botnet Traffic Detection GUI")
root.geometry("950x500")
root.configure(bg="#2e2e2e")  

# Fonts & colors
FONT_TITLE = ("Arial", 16, "bold")
FONT_LABEL = ("Arial", 11)
BG_COLOR = "#2e2e2e"
FG_COLOR = "#f5f5f5"
ENTRY_BG = "#3e3e3e"
ENTRY_FG = "#f5f5f5"


# Layout 

main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

terminal_frame = tk.Frame(main_frame, bg=BG_COLOR)
terminal_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

button_frame = tk.Frame(main_frame, bg=BG_COLOR)
button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)


# Terminal / Log Window
log_text = tk.Text(terminal_frame, state='disabled', width=60, bg="#1e1e1e", fg="#f5f5f5", font=("Consolas", 10))
log_text.pack(fill=tk.BOTH, expand=True)
sys.stdout = RedirectText(log_text)


# Title
title_label = tk.Label(button_frame, text="Botnet Traffic Detection", font=FONT_TITLE, bg=BG_COLOR, fg=FG_COLOR)
title_label.pack(pady=10)

# Button Functions
def show_graphs():
    print("[+] Displaying model performance graphs...")

    # Line plot: Predicted vs Actual (for each model)
    plt.figure(figsize=(10, 5))
    for name, (y_true, y_pred) in trainer.results.items():
        # Count number of predicted and actual labels
        actual_counts = pd.Series(y_true).value_counts().sort_index()
        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        plt.plot(actual_counts.index, actual_counts.values, marker='o', label=f"{name} Actual")
        plt.plot(pred_counts.index, pred_counts.values, marker='x', linestyle='--', label=f"{name} Predicted")

    plt.xlabel("Class (0=Benign, 1=Botnet)")
    plt.ylabel("Count")
    plt.title("Predicted vs Actual Counts for Each Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_confusion_matrix():
    print("\n[+] Confusion Matrices & Classification Reports:")
    for name, (y_true, y_pred) in trainer.results.items():
        print(f"\n--- {name} ---")
        print("Confusion Matrix:")
        print(trainer.confusion_matrix(y_true, y_pred))
        print("Classification Report:")
        print(trainer.classification_report(y_true, y_pred))

def open_prediction_form():
    form_window = tk.Toplevel(root)
    form_window.title("Predict Traffic")
    form_window.geometry("400x700")
    form_window.configure(bg=BG_COLOR)

    entries = {}
    canvas = tk.Canvas(form_window, bg=BG_COLOR)
    scrollbar = tk.Scrollbar(form_window, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg=BG_COLOR)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    for feature in trainer.features:
        lbl = tk.Label(scroll_frame, text=feature, bg=BG_COLOR, fg=FG_COLOR, font=FONT_LABEL)
        lbl.pack(anchor='w', padx=5, pady=2)
        ent = tk.Entry(scroll_frame, bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=FG_COLOR)
        ent.pack(fill=tk.X, padx=5, pady=2)
        entries[feature] = ent

    def predict_traffic():
        input_values = []
        for feature in trainer.features:
            val = entries[feature].get()
            if feature in trainer.categorical_cols:
                input_values.append(val)
            else:
                try:
                    input_values.append(float(val))
                except:
                    input_values.append(0.0)

        X_new = pd.DataFrame([input_values], columns=trainer.features)
        X_new[trainer.categorical_cols] = trainer.encoder.transform(X_new[trainer.categorical_cols].astype(str))

        model = trainer.trained_models["Random Forest"]
        pred_label = model.predict(X_new)
        pred_text = trainer.label_enc.inverse_transform(pred_label)[0]

        messagebox.showinfo("Prediction", f"Predicted Traffic Type: {pred_text}")
        print(f"[+] Predicted Traffic Type: {pred_text}")

    submit_btn = ttk.Button(scroll_frame, text="Submit Prediction", command=predict_traffic)
    submit_btn.pack(pady=10)


# Buttons
style = ttk.Style()
style.configure("TButton", font=("Arial", 11, "bold"), padding=5)

btn_graphs = ttk.Button(button_frame, text="Show Graphs", command=show_graphs)
btn_graphs.pack(pady=5)

btn_cm = ttk.Button(button_frame, text="Show Confusion Matrix", command=show_confusion_matrix)
btn_cm.pack(pady=5)

btn_predict = ttk.Button(button_frame, text="Predict Traffic", command=open_prediction_form)
btn_predict.pack(pady=5)


print("[+] Dark GUI Loaded. Terminal on the left. Buttons on the right. Click 'Predict Traffic' to open the form.")
root.mainloop()
