# ===================== MODEL DEVELOPMENT & PIPELINE EXPORT =====================
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score

# ------------------ Ensure preproc is importable ------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preproc.data_exporter import save_complete_pipeline_outputs
from preproc.config import FILE_PATHS, DATA_LEAKAGE_COLS

# ------------------ LOAD DATA ------------------
train_df = pd.read_csv('marineflow_train.csv')
validate_df = pd.read_csv('marineflow_validation.csv')
test_df = pd.read_csv('marineflow_test.csv')

# ------------------ PREPARE FEATURES & TARGETS ------------------
def prepare_data(df, target_class, target_reg):
    X = df.drop(columns=[target_class, target_reg])
    y_class = df[target_class]
    y_reg = df[target_reg]
    return X, y_class, y_reg

X_train, y_train_class, y_train_reg = prepare_data(train_df, 'demurrage_flag', 'demurrage_amount_usd')
X_val, y_val_class, y_val_reg = prepare_data(validate_df, 'demurrage_flag', 'demurrage_amount_usd')
X_test, y_test_class, y_test_reg = prepare_data(test_df, 'demurrage_flag', 'demurrage_amount_usd')

# ------------------ RANDOM FOREST CLASSIFIER ------------------
rfc = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,              # reduce overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rfc.fit(X_train, y_train_class)
y_pred_class = rfc.predict(X_val)

print("--- CLASSIFICATION METRICS ---")
print("Accuracy:", accuracy_score(y_val_class, y_pred_class))
print("Precision:", precision_score(y_val_class, y_pred_class))
print("Recall:", recall_score(y_val_class, y_pred_class))
print("F1:", f1_score(y_val_class, y_pred_class))

# Cross-validation
cv_scores = cross_val_score(rfc, X_train, y_train_class, cv=5, scoring='f1')
print("CV F1 Scores:", cv_scores)
print("Mean CV F1:", np.mean(cv_scores))

# ------------------ RANDOM FOREST REGRESSOR ------------------
rfr = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rfr.fit(X_train, y_train_reg)
y_pred_reg = rfr.predict(X_val)

print("--- REGRESSION METRICS ---")
print("RMSE:", np.sqrt(mean_squared_error(y_val_reg, y_pred_reg)))
print("MSE:", mean_squared_error(y_val_reg, y_pred_reg))
print("MAE:", mean_absolute_error(y_val_reg, y_pred_reg))

# ------------------ FEATURE IMPORTANCE ------------------
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance_class': rfc.feature_importances_,
    'importance_reg': rfr.feature_importances_
})

# ------------------ ENSURE SAVE DIRECTORIES EXIST ------------------
for path in FILE_PATHS.values():
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ------------------ SAVE COMPLETE PIPELINE ------------------
save_complete_pipeline_outputs(
    X_train, X_val, X_test,
    y_train_class, y_val_class, y_test_class,
    y_train_reg, y_val_reg, y_test_reg,
    scaler=None,  # no scaler
    label_encoders={},  # no label encoders
    feature_names=X_train.columns,
    feature_importance=feature_importance
)
