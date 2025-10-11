from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,r2_score,f1_score,precision_score,recall_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import pandas as pd
import numpy as np


train_df = pd.read_csv('marineflow_train.csv')
validate_df = pd.read_csv('marineflow_validation.csv')
test_df = pd.read_csv('marineflow_test.csv') 

# Leaked Columns 
leaked_cols = [
    'overage_hours_chargeable',
    'laytime_vs_allowed',
    'demurrage_amount_usd',           
    'demurrage_rate_usd_per_day'
]

# CLASSIFICATION DATA 
X_train_class = train_df.drop(columns =[*leaked_cols,'demurrage_flag'])
y_train_class = train_df['demurrage_flag']

X_val_class = validate_df.drop(columns =[*leaked_cols,'demurrage_flag'])
y_val_class = validate_df['demurrage_flag']

X_test_class = test_df.drop(columns =[*leaked_cols,'demurrage_flag'])
y_test_class = test_df['demurrage_flag']


# REGRESSION DATA 
X_train_reg = train_df.drop(columns=[*leaked_cols,'demurrage_amount_usd'])
y_train_reg = train_df['demurrage_amount_usd']

X_val_reg= validate_df.drop(columns=[*leaked_cols,'demurrage_amount_usd'])
y_val_reg = validate_df['demurrage_amount_usd']

X_test_reg = test_df.drop(columns=[*leaked_cols,'demurrage_amount_usd'])
y_test_reg  = test_df['demurrage_amount_usd']


# --------- ADVANCED MODEL DEVELOPMENT ------

# 1. RANDOM FOREST CLASSIFIER
# print('---RANDOM FOREST CLASSIFIER---')

# rfc = RandomForestClassifier(random_state=42,n_jobs=-1)

# rfc.fit(X_train_class,y_train_class)

# y_pred_class = rfc.predict(X_val_class)

# print('Accuracy:',accuracy_score(y_val_class,y_pred_class))
# print('Precision:',precision_score(y_val_class,y_pred_class))
# print('Recall:',recall_score(y_val_class,y_pred_class))
# print('F1:',f1_score(y_val_class,y_pred_class))


# # 2. RANDOM FOREST REGRESSOR 
# print('---RANDOM FOREST REGRESSOR---')

# rfg = RandomForestRegressor(random_state=42,n_jobs=-1)

# rfg.fit(X_train_reg,y_train_reg)

# y_pred_reg =rfg.predict(X_val_reg)

# rmse = np.sqrt(mean_squared_error(y_val_reg,y_pred_reg))

# print('RMSE:',rmse)
# print('MSE:',mean_squared_error(y_val_reg,y_pred_reg))
# print('R2:',r2_score(y_val_reg,y_pred_reg))
# print('MAE:',mean_absolute_error(y_val_reg,y_pred_reg))


# 3. XGBOOST CLASSIFIER 
xgboostclass = XGBClassifier(
    n_estimators = 200,
    random_state = 42,
    max_depth = 3,
    learning_rate = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective = 'binary:logistic')

xgboostclass.fit(X_train_class,y_train_class)

y_pred_cxgb = xgboostclass.predict(X_val_class)

print('Accuracy:',accuracy_score(y_val_class,y_pred_cxgb))
print('Precision:',precision_score(y_val_class,y_pred_cxgb))
print('Recall:',recall_score(y_val_class,y_pred_cxgb))
print('F1:',f1_score(y_val_class,y_pred_cxgb))

# 4. XGBOOST REGRESSOR 

xgboostreg = XGBRegressor(
    n_estimators = 200,
    max_depth = 3,
    random_state = 42,
    learning_rate = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    objective ='reg:squarederror'
    )

xgboostreg.fit(X_train_reg,y_train_reg)

y_pred_rxgb = xgboostreg.predict(X_val_reg)

rmse = np.sqrt(mean_squared_error(y_val_reg,y_pred_rxgb))

print('RMSE:',rmse)
print('MSE:',mean_squared_error(y_val_reg,y_pred_rxgb))
print('R2:',r2_score(y_val_reg,y_pred_rxgb))
print('MAE:',mean_absolute_error(y_val_reg,y_pred_rxgb))






















