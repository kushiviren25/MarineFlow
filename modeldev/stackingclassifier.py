import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from ModelTuned import get_class_model
import warnings
warnings.filterwarnings("ignore")

# Load data
train_df = pd.read_csv('marineflow_train.csv')
test_df = pd.read_csv('marineflow_test.csv')

leaked_cols = [
    'overage_hours_chargeable',
    'laytime_vs_allowed',
    'demurrage_amount_usd',
    'demurrage_rate_usd_per_day'
]

X_train = train_df.drop(columns=[*leaked_cols, 'demurrage_flag'])
y_train = train_df['demurrage_flag']

X_test = test_df.drop(columns=[*leaked_cols, 'demurrage_flag'])
y_test = test_df['demurrage_flag']

# Get Bayesian-tuned base models
best_rfc, best_xgbc, best_gbmc = get_class_model()

base_estimators = [
    ('rfc', best_rfc),
    ('xgbc', best_xgbc),
    ('lgbmc', best_gbmc),
]

#  Logistic Regression as meta-model, wrapped in probability calibration
meta_model = LogisticRegression(max_iter=1000)
calibrated_meta = CalibratedClassifierCV(meta_model, method='sigmoid', cv=5)

# Stacking classifier
stack_classifier = StackingClassifier(
    estimators=base_estimators,
    final_estimator=calibrated_meta,
    cv=5,
    passthrough=True,  # Include base features in meta-model
    n_jobs=-1
)

# Fit stacking classifier
stack_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = stack_classifier.predict(X_test)

print("STACKING METRICS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
