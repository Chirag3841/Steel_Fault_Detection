## Fault Classification in Industrial Machinery using SVM
## Overview

This project focuses on detecting different types of machine faults using sensor data. A Support Vector Machine (SVM) model is built to handle high-dimensional and imbalanced data for accurate multi-class classification.

## Dataset
Samples: 1941
Features: 27 numerical features
Classes (7): Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults

Binary class columns are combined into a single target label (single-label classification).

## Approach
Preprocessing: Handle missing values, merge labels, apply scaling
EDA: Class distribution, correlation heatmap, PCA/t-SNE visualization
Imbalance Handling: class_weight, SMOTE
Model: SVM (RBF / Polynomial kernel)
Tuning: GridSearchCV / RandomizedSearchCV
Evaluation
Accuracy
F1 Score (macro, weighted)
Confusion Matrix

(Optional: Hamming Loss, Subset Accuracy for multi-label)

## Extras
PCA for visualization
LIME / SHAP for interpretability
Comparison with Random Forest, XGBoost, Logistic Regression
Conclusion

Shows that SVM can perform well on imbalanced industrial datasets with proper preprocessing and tuning.retability, and computation time

