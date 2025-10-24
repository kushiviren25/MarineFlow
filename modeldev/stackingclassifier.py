from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier
from modeldev.TunedClassifier import classifier_model
from sklearn.calibration import CalibratedClassifierCV
import warnings
import logging

# Silence warnings and logs
# warnings.filterwarnings("ignore")
# logging.getLogger("lightgbm").setLevel(logging.CRITICAL)
# logging.getLogger("xgboost").setLevel(logging.CRITICAL)

# Load your data
import pandas as pd
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

# Get tuned models
best_rfc, best_xgbc, best_gbmc = classifier_model()

# Force silent base models
best_rfc.set_params(verbose=0)
best_xgbc.set_params(verbosity=0)
best_gbmc.set_params(verbose=-1)

estimator = [
    ('rfc', best_rfc),
    ('xgbc', best_xgbc),
    ('gbmc', best_gbmc)
    ]
meta_model = LogisticRegression(max_iter=1000)

calibratedclass = CalibratedClassifierCV(
    meta_model,cv = 5, method='sigmoid')

# Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators= estimator,
    final_estimator=calibratedclass,
    cv=5,
    n_jobs=-1,
    passthrough=True,
    verbose=0
)

stacking_classifier.fit(X_train, y_train)

# Predictions
y_pred = stacking_classifier.predict(X_test)

# Only evaluation metrics
print('----EVALUATION METRICS FOR STACKING CLASSIFIER---')
print('Accuracy :', round(accuracy_score(y_test, y_pred), 2))
print('Precision :', round(precision_score(y_test, y_pred), 2))
print('Recall :', round(recall_score(y_test, y_pred), 2))
print('F1 :', round(f1_score(y_test, y_pred), 2))
