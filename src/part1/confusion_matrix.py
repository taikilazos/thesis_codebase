import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global font scale and style for seaborn
plt.rcParams['font.weight'] = 'bold'  # Make all fonts bold
sns.set(font_scale=1.5)  # Increase all fonts by 50%

# Your confusion matrix data
classes = ['O', 'GOOGLE_EASY', 'MEDICAL_ABBR', 'MEDICAL_NAME', 
           'GENERAL_COMPLEX', 'GENERAL_ABBR', 'GOOGLE_HARD', 'MULTISENSE']

# Create the matrix data
data = np.array([
    [25016, 662, 18, 72, 106, 6, 8, 1],
    [270, 3260, 54, 237, 24, 0, 90, 4],
    [6, 28, 849, 2, 0, 7, 41, 0],
    [29, 37, 53, 319, 1, 0, 16, 0],
    [142, 37, 0, 0, 307, 3, 0, 0],
    [6, 6, 15, 0, 0, 103, 0, 0],
    [57, 624, 33, 0, 3, 0, 461, 0],
    [1, 26, 0, 0, 1, 0, 0, 0]
])

# Normalize by row (true class)
matrix = data / data.sum(axis=1, keepdims=True)

# Create larger figure
plt.figure(figsize=(12, 10))

# Create heatmap with larger and bold fonts
sns.heatmap(matrix, 
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes,
            annot_kws={'size': 12, 'weight': 'bold'})  # Added bold weight

# Customize fonts and labels with larger sizes and bold weight
plt.title('Confusion Matrix (Normalized by True Class)', 
          fontsize=16, pad=20, weight='bold')
plt.xlabel('Predicted Label', 
          fontsize=14, labelpad=15, weight='bold')
plt.ylabel('True Label', 
          fontsize=14, labelpad=15, weight='bold')

# Rotate x-axis labels with larger bold font
plt.xticks(rotation=45, ha='right', fontsize=12, weight='bold')
plt.yticks(rotation=0, fontsize=12, weight='bold')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save with high DPI for quality
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
