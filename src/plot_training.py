# Emily Proctor

import matplotlib.pyplot as plt
import pandas as pd
import argparse

# arguments
parser = argparse.ArgumentParser()
training_file = parser.add_argument('training_file', type=str, help='Path to the training log file')
output_folder = parser.add_argument('output_folder', type=str, help='Path to the output folder for plots')
args = parser.parse_args()

training_file = args.training_file
output_folder = args.output_folder

# read training log
df = pd.read_csv(args.training_file, sep="\t", skiprows=2)

metrics = {
    "loss": "loss",
    "balacc": "balanced accuracy",
    "prec": "precision",
    "recall": "recall",
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10)) # 2x2 grid
axes = axes.ravel() # flatten 2D to 1D array for iteration

for ax, (metric, label) in zip(axes, metrics.items()): # iterate through metrics

    # get columns
    train_col = f"train_{metric}"
    val_col   = f"val_{metric}"

    # plot training and validation curves
    ax.plot(df["epoch"], df[train_col], label="training set", linewidth=2, color='steelblue', marker='o', markersize=4) 
    ax.plot(df["epoch"], df[val_col], label="validation set", linewidth=2, color='mediumseagreen', marker='o', markersize=4)

    # set labels and title
    ax.set_title(label, fontsize=15)
    ax.set_xlabel("epoch", fontsize=15)
    ax.set_ylabel(label, fontsize=15)
    ax.grid(alpha=0.3)
    ax.legend()

fig.tight_layout()

# save figure
plt.savefig(f"{output_folder}/train_val_curves.pdf", format="pdf", dpi=300)
