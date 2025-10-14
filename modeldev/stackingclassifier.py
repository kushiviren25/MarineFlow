from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd 
from ModelTuned import get_class_model

# Suppress printouts from GridSearchCV
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('marineflow_train.csv')
test_df = pd.read_csv('marineflow_test.csv') 

 
leaked_cols = [
    'overage_hours_chargeable',
    'laytime_vs_allowed',
    'demurrage_amount_usd',           
    'demurrage_rate_usd_per_day'
]

# Training data 
X_train_class = train_df.drop(columns=[*leaked_cols,'demurrage_flag'])
y_train_class = train_df['demurrage_flag']

# Testing data 
X_test_class = test_df.drop(columns=[*leaked_cols,'demurrage_flag'])
y_test_class = test_df['demurrage_flag']


best_rfc,best_xgbc,best_gbmc = get_class_model()

base_estimators = [
    ('rfc',best_rfc),
    ('xgbc',best_xgbc),
    ('lgbmc',best_gbmc),
]

meta_model = LogisticRegression()

stack_classifier = StackingClassifier(
    estimators= base_estimators,
    final_estimator= meta_model,
    cv = 5
)

stack_classifier.fit(X_train_class,y_train_class)

y_pred = stack_classifier.predict(X_test_class)

print("STACKING METRICS")
print("Accuracy:", accuracy_score(y_test_class, y_pred))
print("Precision:", precision_score(y_test_class, y_pred))
print("Recall:", recall_score(y_test_class, y_pred))
print("F1 Score:", f1_score(y_test_class, y_pred))