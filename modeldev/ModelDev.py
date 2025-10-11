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
from preproc.feature_builder import run_complete_feature_engineering

# ------------------ TIMESTAMP PREPROCESSING ------------------
def preprocess_timestamps(df):
    """
    Ensure required timestamp columns exist and are datetime.
    Maps common alternative column names to expected ones.
    """
    timestamp_mapping = {
        'depart_ts': ['depart_ts', 'departure_time', 'departure_date'],
        'arrival_ts': ['arrival_ts', 'arrival_time', 'arrival_date']
    }

    for expected_col, possible_cols in timestamp_mapping.items():
        for col in possible_cols:
            if col in df.columns:
                df[expected_col] = pd.to_datetime(df[col], errors='coerce')
                break
        else:
            # If none of the possible columns exist, raise an error
            raise ValueError(f"Missing required timestamp column: {expected_col}")

    return df

# ------------------ LOAD DATA ------------------
train_df = pd.read_csv('marineflow_train.csv')
validate_df = pd.read_csv('marineflow_validation.csv')
test_df = pd.read_csv('marineflow_test.csv')

# ------------------ APPLY TIMESTAMP PREPROCESSING ------------------
train_df = preprocess_timestamps(train_df)
validate_df = preprocess_timestamps(validate_df)
test_df = preprocess_timestamps(test_df)

# ------------------ COMPLETE FEATURE ENGINEERING ------------------
full_df = pd.concat([train_df, validate_df, test_df], axis=0).reset_index(drop=True)
(X_train_scaled, X_val_scaled, X_test_scaled, 
 y_class_train, y_class_val, y_class_test,
 y_reg_train, y_reg_val, y_reg_test,
 scaler, label_encoders, feature_names, feature_importance) = run_complete_feature_engineering(full_df)

# ------------------ RANDOM FOREST CLASSIFIER ------------------
rfc = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,              # reduce overfitting
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rfc.fit(X_train_scaled, y_class_train)
y_pred_class = rfc.predict(X_val_scaled)

print("--- CLASSIFICATION METRICS ---")
print("Accuracy:", accuracy_score(y_class_val, y_pred_class))
print("Precision:", precision_score(y_class_val, y_pred_class))
print("Recall:", recall_score(y_class_val, y_pred_class))
print("F1:", f1_score(y_class_val, y_pred_class))

cv_scores = cross_val_score(rfc, X_train_scaled, y_class_train, cv=5, scoring='f1')
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
rfr.fit(X_train_scaled, y_reg_train)
y_pred_reg = rfr.predict(X_val_scaled)

print("--- REGRESSION METRICS ---")
print("RMSE:", np.sqrt(mean_squared_error(y_reg_val, y_pred_reg)))
print("MSE:", mean_squared_error(y_reg_val, y_pred_reg))
print("MAE:", mean_absolute_error(y_reg_val, y_pred_reg))

# ------------------ FEATURE IMPORTANCE ------------------
feature_importance['importance_class'] = rfc.feature_importances_
feature_importance['importance_reg'] = rfr.feature_importances_

# ------------------ ENSURE SAVE DIRECTORIES EXIST ------------------
for path in FILE_PATHS.values():
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ------------------ SAVE COMPLETE PIPELINE ------------------
save_complete_pipeline_outputs(
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_class_train, y_class_val, y_class_test,
    y_reg_train, y_reg_val, y_reg_test,
    scaler=scaler,
    label_encoders=label_encoders,
    feature_names=feature_names,
    feature_importance=feature_importance
)
