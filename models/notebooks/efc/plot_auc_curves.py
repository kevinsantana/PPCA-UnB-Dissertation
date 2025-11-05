import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# This script demonstrates how to plot ROC curves for multiple experiments.
# You will need to replace the dummy data below with your actual experimental results.

# --- 1. GATHER YOUR DATA ---
# For each experimental setting you want to compare, you need two things:
#   a) The true labels of your test set (y_true).
#   b) The prediction scores from your model for each sample in the test set (y_scores).
#      - These are NOT the final 0/1 predictions.
#      - They are the continuous output values before a threshold is applied.
#      - For EFC, these are the 'energy' scores.
#      - For other classifiers, this is often the output of `predict_proba()[:, 1]`.

# --- 2. REPLACE DUMMY DATA WITH YOURS ---
# Example: Let's assume you have one test set and three different models/settings.
# The `y_true` would be the same for all if you use the same test set.

# This should be your actual test set's true labels (0 for licit, 1 for illicit)
# This is just dummy data for demonstration (50 negative, 50 positive samples).
y_true = np.concatenate([np.zeros(50), np.ones(50)])

# Replace these with the scores from your actual experiments.
# The keys are the names that will appear in the plot legend.
# The values are the corresponding prediction scores.
# I've created dummy scores that will result in different AUC values for illustration.
experiment_scores = {
    "Experiment 3: Combined FS + SMOTE (k=30)": np.concatenate([np.random.rand(50) * 0.4, np.random.rand(50) * 0.6 + 0.3]),
    "Experiment 4: Your Setting 1": np.concatenate([np.random.rand(50) * 0.5, np.random.rand(50) * 0.5 + 0.25]),
    "Experiment 5: Your Setting 2": np.concatenate([np.random.rand(50) * 0.6, np.random.rand(50) * 0.4 + 0.35]),
}

# --- 3. CALCULATE ROC CURVES ---
# This part of the script calculates the FPR, TPR, and AUC for each experiment.
roc_results = {}
for name, scores in experiment_scores.items():
    # For EFC, higher energy scores typically indicate the positive (illicit) class.
    # Ensure your scores are aligned so that higher values indicate the positive class.
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    roc_results[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

# --- 4. PLOT THE RESULTS ---
# This part of the script generates the plot.
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 8))

# Using colors from your project's context for consistency
colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231']
for i, (name, data) in enumerate(roc_results.items()):
    ax.plot(data['fpr'], data['tpr'], color=colors[i % len(colors)], lw=2,
            label=f"{name} (AUC = {data['auc']:.3f})")

# Plot the "no-skill" line (random guessing)
ax.plot([0, 1], [0, 1], 'k--', lw=2, label='No-Skill (AUC = 0.500)')

# --- 5. FORMAT AND SHOW THE PLOT ---
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('Comparison of ROC Curves for Different Experimental Settings', fontsize=16)
ax.legend(loc="lower right", fontsize=12)

fig.savefig('roc_curves_comparison.png', dpi=400)
plt.show()

# To save the figure to a file, uncomment the following line:
# fig.savefig('roc_curves_comparison.png', dpi=