# Fault Classification in Industrial Machinery using SVM  

## Overview  
This project focuses on detecting different types of machine faults using sensor data. A Support Vector Machine (SVM) model is built to handle high-dimensional and imbalanced data for accurate multi-class classification.

---

## Dataset  
- **Samples:** 1941  
- **Features:** 27 numerical features  
- **Classes (7):**  
  - Pastry  
  - Z_Scratch  
  - K_Scatch  
  - Stains  
  - Dirtiness  
  - Bumps  
  - Other_Faults  

Binary class columns are combined into a single target label (single-label classification).

---

## Approach  

- **Preprocessing:**  
  - Handle missing values  
  - Merge binary labels  
  - Feature scaling (StandardScaler / MinMaxScaler)  

- **EDA:**  
  - Class distribution  
  - Correlation heatmap  
  - PCA / t-SNE visualization  

- **Imbalance Handling:**  
  - `class_weight='balanced'`  
  - SMOTE  

- **Model:**  
  - SVM (RBF / Polynomial kernel)  

- **Hyperparameter Tuning:**  
  - GridSearchCV / RandomizedSearchCV  

---

## Evaluation  

- Accuracy  
- F1 Score (macro, weighted)  
- Confusion Matrix  

(Optional for multi-label: Hamming Loss, Subset Accuracy)

---

## Additional Work  

- PCA for visualization  
- LIME / SHAP for interpretability  
- Comparison with Random Forest, XGBoost, Logistic Regression  

---

## Conclusion  

This project demonstrates that SVM can effectively handle imbalanced industrial datasets when combined with proper preprocessing and tuning.
