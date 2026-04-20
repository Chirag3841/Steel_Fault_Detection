## Steel  Fault Detection using Machine Learning

This project implements a multi-class fault classification system for detecting surface defects in steel plates using machine learning models. It evaluates multiple algorithms and an ensemble approach to achieve robust performance on industrial data.

## Overview

The goal of this project is to build a reliable and scalable defect detection system using sensor-based features.
Multiple models are trained and compared to handle high-dimensional data, class imbalance, and multi-class classification. A stacking classifier is also implemented to combine model strengths.

## Tech Stack
Models:
  * Support Vector Machine (SVM)
  * Random Forest
  * XGBoost
  * Stacking Classifier (Ensemble)

Libraries:
   * Scikit-learn
   * Pandas
   * NumPy
   * Matplotlib
   * Seaborn
   * Imbalance Handling: SMOTE
   * Deployment: Streamlit + Pickle

Dataset:
   * Samples: 1941
   * Features: 27 numerical features

Classes (7):
   * Pastry
   * Z_Scratch
   * K_Scatch
   * Stains
   * Dirtiness
   * Bumps
   * Other_Faults

Multi-label columns are converted into a single-label classification problem.

## Key Features
   * Multi-class classification using multiple ML models
   * Model comparison and performance evaluation
   * Ensemble learning using stacking classifier
   * Handles class imbalance using SMOTE and class weights
   * Hyperparameter tuning using GridSearchCV
   * Model saving for real-time prediction
## Approach
Data Preprocessing
  * Missing value handling
  * Multi-label to single-label conversion
  * Feature scaling
Exploratory Data Analysis
  * Class distribution analysis
  * Correlation heatmap
  * PCA and t-SNE visualization
Model Training
  * Support Vector Machine (RBF kernel)
  * Random Forest
  * XGBoost
Hyperparameter Tuning
  * GridSearchCV with 10-fold cross-validation

## Best parameters:

* SVM: C=10, gamma=0.1, kernel=rbf
* Random Forest: n_estimators=200, max_depth=None
* XGBoost: learning_rate=0.1, max_depth=5, n_estimators=200

## Results
Individual models
| Model         | Accuracy | Weighted F1 | Hamming Loss |
| ------------- | -------- | ----------- | ------------ |
| SVM           | 0.7609   | 0.7604      | 0.2391       |
| Random Forest | 0.8021   | 0.8018      | 0.1979       |
| XGBoost       | 0.7969   | 0.7975      | 0.2031       |

Ensemble model
| Model               | Accuracy | Weighted F1 | Hamming Loss |
| ------------------- | -------- | ----------- | ------------ |
| Stacking Classifier | 0.7969   | 0.80        | 0.2031       |

## Key Insights
  * Random Forest achieved the highest individual accuracy
  * XGBoost provided balanced performance across classes
  * SVM performed well after tuning but was slightly lower than ensemble methods
  * Stacking classifier improved overall robustness by combining models
  * SMOTE significantly improved minority class prediction

## Project Structure
```
├── SteelFault_Detection.ipynb
├── steel_fault_detection.py
├── app.py
├── model.pkl
├── scaler.pkl
├── steel_plates_faults.csv
├── README.md
```

## Deployment
1. Install dependencies
 ```
pip install -r requirements.txt
```
2. Run the application
```
streamlit run app.py
```

## Conclusion

This project demonstrates how multiple machine learning models along with ensemble techniques can effectively solve industrial fault detection problems. Random Forest and XGBoost performed strongly, while stacking improved overall robustness.

## Author 
```
Chirag Sharma
MSIT
```
