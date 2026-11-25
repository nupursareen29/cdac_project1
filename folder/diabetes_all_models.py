# Compare parametric and non-parametric classification models on diabetes.csv
# Install if needed: pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier

# 1. Load data
df = pd.read_csv("diabetes.csv")
print("Dataset shape:", df.shape)
print(df.head())
print("\\nTarget distribution:")
print(df["Outcome"].value_counts())

# 2. Separate predictors and target
X = df.drop(columns="Outcome")
y = df["Outcome"]

# In this dataset, zero values in these fields commonly represent missing measurements.
# Replace those zeros with NaN, then impute using the median learned from training data.
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[zero_as_missing] = X[zero_as_missing].replace(0, np.nan)

# 3. Train/test split: keep the class proportions similar in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Models: parametric and non-parametric / flexible models
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
    "Gaussian Naive Bayes": GaussianNB(),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, min_samples_leaf=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=42, min_samples_leaf=3
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=300, random_state=42, min_samples_leaf=3
    ),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(50, 25), max_iter=1500,
        random_state=42, early_stopping=True
    ),
}

# Scaling is important for distance/gradient-based models; trees do not need it.
scaled_models = {
    "Logistic Regression", "LDA", "QDA", "Gaussian Naive Bayes",
    "KNN (k=7)", "SVM (RBF)", "MLP Neural Network"
}

results = []
fitted_models = {}

# 5. Train and evaluate each model on the same held-out test set
for name, estimator in models.items():
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if name in scaled_models:
        steps.append(("scaler", StandardScaler()))
    pipe = Pipeline(steps + [("model", estimator)])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    probabilities = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    })
    fitted_models[name] = pipe

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\\nModel comparison (sorted by ROC-AUC):")
print(results_df.round(3).to_string(index=False))
results_df.to_csv("diabetes_model_comparison.csv", index=False)

# 6. Inspect the top-ranked model on this split
best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]
best_predictions = best_model.predict(X_test)
print(f"\\nTop model by ROC-AUC on this split: {best_name}")
print("\\nConfusion matrix (rows=true class, columns=predicted class):")
print(confusion_matrix(y_test, best_predictions))
print("\\nClassification report:")
print(classification_report(y_test, best_predictions, zero_division=0))

print("\\nNote: these results are a baseline on one train/test split, not a clinical diagnosis.")
