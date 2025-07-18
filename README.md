Topic- Fault Classification in Industrial Machinery using SVM
 Problem Statement-
In predictive maintenance, early and accurate identification of faulty machine components is crucial to minimize downtime, reduce maintenance costs, and ensure operational safety.

This project focuses on building a Support Vector Machine (SVM) model for multi-class fault classification from high-dimensional and imbalanced sensor data.

 Dataset Overview-
Samples: 1941

Features: 27 numerical (sensor values, geometric parameters, etc.)

Target Classes (7 types):

Pastry

Z_Scratch

K_Scatch

Stains

Dirtiness

Bumps

Other_Faults

 Each class is a binary column, but assume single-label classification unless otherwise specified.

 Project Pipeline-
1. Data Preprocessing
Combine binary target columns into a single label.

Handle missing/null values.

Feature scaling using StandardScaler or MinMaxScaler.

2. Exploratory Data Analysis (EDA)
Plot class distribution to assess imbalance.

Correlation heatmap of features.

Visualize separability with PCA or t-SNE.

3. Handling Class Imbalance
Use class_weight='balanced' in SVM.

Apply SMOTE or a combination of over/under-sampling techniques.

4. Model Building: SVM
Use sklearn.svm.SVC with:

Kernel: RBF / Polynomial

Hyperparameter Tuning: GridSearchCV or RandomizedSearchCV

Class weights to address imbalance

5. Evaluation Metrics
Accuracy

F1 Score (macro and weighted)

Confusion Matrix & Classification Report

For multi-label (if applied): Hamming Loss, Subset Accuracy

6. Model Interpretability
Use PCA for 2D decision boundary plots

Apply LIME or SHAP (via wrappers) to understand feature influence

7. (Optional) Comparative Analysis
Compare SVM with:

Random Forest

XGBoost

Logistic Regression

Analyze trade-offs in accuracy, interpretability, and computation time

