# AI600 - Deep Learning - Assignment 2
### The "Quick, Draw!" Challenge

This repository contains the final submission files for the AI-600 PA2 mapping to the Quick, Draw! Challenge classification problem. 

The goal is to accurately classify 15 classes of drawings using strictly Core Multi-Layer Perceptrons (MLPs), balancing Width vs. Depth while avoiding Convolutional properties.

## Contents:
- `25280040_PA2.ipynb`: The primary Jupyter Notebook containing the full training pipeline, Data Augmentation passes, Loss/Accuracy Visualizations, and Confusion Matrices for the Wide (Pancake), Deep (Tower), and Optimized (Champion) models.
- `inference.py`: A standalone inference script embedding the final `ChampionMLP` PyTorch class and a `predict_image()` execution loop.
- `champion_weights.pth`: The final trained weights for the 60-epoch Champion model (approximating 85.6% validation accuracy).
- `submission.txt`: The corresponding Leaderboard CSV predictions.
- `report.tex` and `*.png`: The formal LaTeX report documenting the theoretical analysis and the extracted high-resolution graphics natively embedded within the document.
