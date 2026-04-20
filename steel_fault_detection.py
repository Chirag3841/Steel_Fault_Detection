# ─── Imports ────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, hamming_loss, confusion_matrix, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ─── Data Loading ───────────────────────────────────────────────────────────────
DATA_PATH = 'C:\\Github_Projects\\steel_fault_detection\\steel_plates_faults.csv'   # update path if needed
d = pd.read_csv(DATA_PATH)

# ─── Data Understanding ─────────────────────────────────────────────────────────
print(d.head())
print(d.shape)
print(d.describe().T)
print(d.columns.tolist())
print(d.isnull().sum())
print(d.corr(numeric_only=True))
d.info()
print(d.dtypes)

# ─── Label Mapping ──────────────────────────────────────────────────────────────
mapping = {
    'Pastry': 0, 'Dirtiness': 1, 'K_Scatch': 2,
    'Bumps': 3, 'Other_Faults': 4, 'Stains': 5, 'Z_Scratch': 6
}
reverse_mapping = {v: k for k, v in mapping.items()}

# ─── Preprocessing ──────────────────────────────────────────────────────────────
d.drop_duplicates(inplace=True)

# Combine binary fault columns into a single target column
fault_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
d['Fault_Types'] = d[fault_cols].idxmax(axis=1).map(mapping)
d.drop(columns=fault_cols, inplace=True)

# Drop unnecessary columns
d.drop(columns=[
    'Square_Index', 'Sum_of_Luminosity', 'X_Minimum', 'X_Perimeter',
    'SigmoidOfAreas', 'Edges_X_Index', 'Y_Minimum', 'Y_Maximum'
], inplace=True)

print(d.shape)
print(d.columns.tolist())

# Add human-readable fault label
d['Fault_Label'] = d['Fault_Types'].map(reverse_mapping).astype(str)

# ─── EDA ────────────────────────────────────────────────────────────────────────
# Steel type distribution
steel_cols = ['TypeOfSteel_A300', 'TypeOfSteel_A400']
d[steel_cols].sum().plot(kind='bar', color='coral')
plt.title("Distribution of Steel Types")
plt.xticks(rotation=0)
plt.show()

# Pairplot
sns.pairplot(
    d[['Pixels_Areas', 'Edges_Index', 'Luminosity_Index', 'Orientation_Index', 'Fault_Label']],
    hue='Fault_Label'
)
plt.show()

# Boxplots for all numeric columns
numeric_cols = d.select_dtypes(include=np.number).columns.difference(fault_cols)
plt.figure(figsize=(20, 30))
for i, col in enumerate(numeric_cols):
    plt.subplot(9, 4, i + 1)
    sns.boxplot(y=d[col], color="lightgreen")
    plt.title(col)
plt.tight_layout()
plt.show()

# Histograms
d.select_dtypes(include='number').hist(bins=30, figsize=(20, 15), color='skyblue')
plt.suptitle("Histograms of Numerical Features", fontsize=20)
plt.show()

# ─── Feature / Target Split ─────────────────────────────────────────────────────
X = d.drop(columns=['Fault_Types', 'Fault_Label'])
y = d['Fault_Types']

# Class distribution (after y is defined)
plt.figure(figsize=(10, 6))
sns.countplot(x=y.map(reverse_mapping))
plt.title('Class Distribution of Fault Types')
plt.xlabel('Fault Type')
plt.xticks(rotation=45)
plt.show()

# ─── Outlier Clipping ───────────────────────────────────────────────────────────
for col in X.columns:
    q1, q3 = X[col].quantile([0.01, 0.99])
    X[col] = X[col].clip(q1, q3)

# Fill nulls if any
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.mean())

# ─── Scaling ────────────────────────────────────────────────────────────────────
s = StandardScaler()
X_scaled = s.fit_transform(X)

# Correlation heatmap (post-scaling)
plt.figure(figsize=(30, 15))
correlation_matrix = np.corrcoef(X_scaled.T)
sns.heatmap(correlation_matrix, annot=True,
            xticklabels=X.columns, yticklabels=X.columns, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# PCA 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 5))
for fault_id in y.unique():
    mask = y.values == fault_id
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=reverse_mapping[fault_id], alpha=0.6)
plt.title('PCA Visualization of Fault Classes')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend()
plt.show()

# ─── Train/Test Split ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ─── SMOTE (handle class imbalance) ─────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_val, y_val = smote.fit_resample(X_train, y_train)
print("After SMOTE:", X_val.shape)

# ─── Model Definitions ──────────────────────────────────────────────────────────
models = {
    'SVM':          OneVsRestClassifier(SVC(class_weight='balanced', probability=True, random_state=42)),
    'Random Forest':OneVsRestClassifier(RandomForestClassifier(class_weight='balanced', random_state=42)),
    'XGBoost':      OneVsRestClassifier(XGBClassifier(eval_metric='logloss', random_state=42)),
}

param_grids = {
    'SVM': {
        'estimator__C':      [0.1, 1, 10],
        'estimator__gamma':  [0.01, 0.1],
        'estimator__kernel': ['rbf']
    },
    'Random Forest': {
        'estimator__n_estimators':    [100, 200],
        'estimator__max_depth':       [10, None],
        'estimator__min_samples_split':[2, 5]
    },
    'XGBoost': {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth':    [3, 5],
        'estimator__learning_rate':[0.01, 0.05, 0.1]
    },
}

# ─── GridSearchCV Tuning ────────────────────────────────────────────────────────
best_model = {}
for name, model in models.items():
    g = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1
    )
    g.fit(X_val, y_val)
    best_model[name] = g.best_estimator_
    print(f'Best params  [{name}]: {g.best_params_}')
    print(f'Best F1-macro[{name}]: {g.best_score_:.4f}')

# ─── Individual Model Evaluation ────────────────────────────────────────────────
for name, model in best_model.items():
    y_pred = model.predict(X_test)
    print(f'\n{name} Accuracy       : {accuracy_score(y_test, y_pred):.4f}')
    print(f'{name} F1 (weighted)  : {f1_score(y_test, y_pred, average="weighted"):.4f}')
    print(f'{name} Hamming Loss   : {hamming_loss(y_test, y_pred):.4f}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')

# ─── Stacking Classifier ────────────────────────────────────────────────────────
stacking_clf = StackingClassifier(estimators=[
    ('svm', SVC(
        probability=True,
        C=best_model['SVM'].estimator.C,
        gamma=best_model['SVM'].estimator.gamma,
        kernel=best_model['SVM'].estimator.kernel,
        class_weight='balanced', random_state=42
    )),
    ('rf', RandomForestClassifier(
        n_estimators=best_model['Random Forest'].estimator.n_estimators,
        max_depth=best_model['Random Forest'].estimator.max_depth,
        min_samples_split=best_model['Random Forest'].estimator.min_samples_split,
        class_weight='balanced', random_state=42
    )),
    ('xgb', XGBClassifier(
        n_estimators=best_model['XGBoost'].estimator.n_estimators,
        max_depth=best_model['XGBoost'].estimator.max_depth,
        learning_rate=best_model['XGBoost'].estimator.learning_rate,
        subsample=best_model['XGBoost'].estimator.subsample,
        eval_metric='logloss', random_state=42
    ))
])
stacking_clf.fit(X_val, y_val)
y_pred_stacking = stacking_clf.predict(X_test)

print(f'\nStacking Hamming Loss   : {hamming_loss(y_test, y_pred_stacking):.4f}')
print(f'Stacking Accuracy       : {accuracy_score(y_test, y_pred_stacking):.4f}')
print(f'Classification Report:\n{classification_report(y_test, y_pred_stacking, target_names=fault_cols)}')

# Confusion matrix heatmap
plt.figure(figsize=(10, 8))
cm_stacking = confusion_matrix(y_test, y_pred_stacking)
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues',
            xticklabels=fault_cols, yticklabels=fault_cols)
plt.title('Confusion Matrix — Stacking Classifier')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ─── Save Model & Scaler ────────────────────────────────────────────────────────
joblib.dump(stacking_clf, 'model.pkl')
joblib.dump(s, 'scaler.pkl')
print('model.pkl and scaler.pkl saved successfully!')

# ─── Interpretability: SVM Decision Boundaries (PCA 2D) ─────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
svm_2d = SVC(
    kernel=best_model['SVM'].estimator.kernel,
    C=best_model['SVM'].estimator.C,
    gamma=best_model['SVM'].estimator.gamma,
    class_weight='balanced', random_state=42
)
svm_2d.fit(X_pca, y)
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array([np.where(svm_2d.classes_ == z)[0][0] for z in Z]).reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=[np.where(svm_2d.classes_ == z)[0][0] for z in y],
            cmap='viridis', alpha=0.8)
plt.title('SVM Decision Boundaries (PCA 2D)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.show()

# ─── Interpretability: LIME ─────────────────────────────────────────────────────
from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(
    X_train, feature_names=X.columns.tolist(),
    class_names=fault_cols, mode='classification'
)
for idx in range(3):
    exp = explainer.explain_instance(X_test[idx], best_model['SVM'].predict_proba, num_features=10)
    fig = exp.as_pyplot_figure()
    plt.title(f'LIME Explanation — Test Sample {idx + 1}')
    plt.show()

# ─── Submission File ────────────────────────────────────────────────────────────
y_pred = y_pred_stacking
submission = pd.DataFrame({
    'Id': range(len(y_pred)),
    'Predicted': y_pred
})
submission['Predicted_Label'] = submission['Predicted'].map(reverse_mapping)
submission.to_csv('submission.csv', index=False)
print(submission.shape)
print(submission.head())

submission['Predicted_Label'].value_counts().plot(kind='bar')
plt.title('Predicted Fault Distribution')
plt.xlabel('Fault Type')
plt.ylabel('Count')
plt.show()
