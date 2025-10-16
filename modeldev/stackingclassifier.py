import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ModelTuned import get_class_model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
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

# Call the models via model tuned function 
best_rfc,best_xgbc,best_gbmc = get_class_model()

# Setup Base Estimators  
base_estimator  = [
    ('rfc',best_rfc),
    ('xgbc',best_xgbc),
    ('gbmc',best_gbmc),
]

# Set the Meta Model 
meta_model = LogisticRegression(max_iter=1000)

# Optimise better with Calibration
calibrated_model = CalibratedClassifierCV(meta_model,cv = 5,method ='sigmoid')

# Setup the Stacking Classifier 
stacking_classifier = StackingClassifier(
    estimators=base_estimator,
    final_estimator=calibrated_model,
    cv =5,
    n_jobs=-1,
    passthrough=True
) 

# Fit on training data 
stacking_classifier.fit(X_train,y_train)

# Predictions on testing data
y_test_class = stacking_classifier.predict(X_test)

# Evaluation Metrics 
print('Accuracy ;',accuracy_score(y_test,y_test_class))
print('Precision ;',precision_score(y_test,y_test_class))
print('Recall ;',recall_score(y_test,y_test_class))
print('F1  ;',f1_score(y_test,y_test_class))

