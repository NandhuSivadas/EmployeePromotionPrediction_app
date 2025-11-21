import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# --------------------------
# 1️⃣ Load the cleaned dataset (before SMOTE)
# --------------------------
data = pd.read_csv("Dataset/employee_promotion_cleaned_before_smote.csv")
print("Data Loaded. Shape:", data.shape)

# --------------------------
# 2️⃣ Separate features and target
# --------------------------
X = data.drop('is_promoted', axis=1)
y = data['is_promoted']

# --------------------------
# 3️⃣ Apply SMOTE to the entire dataset
# --------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("✅ After SMOTE - Dataset shape:", X_res.shape)
print("Class distribution after SMOTE:\n", y_res.value_counts())

# --------------------------
# 4️⃣ Split 20% for testing
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Save the test set
test_data = pd.concat([X_test, y_test], axis=1)
os.makedirs("Dataset", exist_ok=True)
test_data.to_csv("Dataset/test.csv", index=False)
print("✅ Test dataset (20%) saved as 'Dataset/test.csv'")

# --------------------------
# 5️ Train Random Forest and XGBoost on SMOTEd training data
# --------------------------
rf_model = RandomForestClassifier(min_samples_leaf=10, min_samples_split=20, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# --------------------------
# 6️ Evaluate training performance with metrics + confusion matrix
# --------------------------
def training_metrics_with_cm(model, X_train, y_train, model_name):
    y_pred = model.predict(X_train)
    y_prob = model.predict_proba(X_train)[:, 1]
    
    acc = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    roc = roc_auc_score(y_train, y_prob)
    prec = precision_score(y_train, y_pred)
    
    print(f"\n===== {model_name} Training Metrics =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"Precision: {prec:.4f}")
    
    cm = confusion_matrix(y_train, y_pred)
    print("\nConfusion Matrix:\n", cm)
    light_colors = ListedColormap(['#FFFFFF', '#D0F0C0', '#A8E6A8'])
    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(cmap=light_colors)
    plt.title(f"{model_name} - Training Confusion Matrix")
    plt.show()
    
    return acc, f1, roc, prec

# Compute metrics for both models
rf_train_metrics = training_metrics_with_cm(rf_model, X_train, y_train, "Random Forest")
xgb_train_metrics = training_metrics_with_cm(xgb_model, X_train, y_train, "XGBoost")

# --------------------------
# 7️⃣ Save trained models
# --------------------------
MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
print(f"\n✅ Models saved in folder: {MODEL_DIR}")

# --------------------------
# 8️⃣ Training Model Comparison
# --------------------------
comparison_train = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [rf_train_metrics[0], xgb_train_metrics[0]],
    "F1-score": [rf_train_metrics[1], xgb_train_metrics[1]],
    "ROC-AUC": [rf_train_metrics[2], xgb_train_metrics[2]],
    "Precision": [rf_train_metrics[3], xgb_train_metrics[3]]
})

print("\n✅ Training Model Comparison:\n", comparison_train)
