from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,r2_score,f1_score,precision_score,recall_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
import joblib
import pandas as pd
import numpy as np

# Read CSV files
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

#  PHASE 1 
# ------BASELINE MODEL DEVELOPMENT -----
# 1. LOGISTIC REGRESSION 
print('LOGISTIIC REGRESSION')
logreg = LogisticRegression(C=1.0,solver='lbfgs',max_iter=100)

logreg.fit(X_train_class,y_train_class)
y_pred_log = logreg.predict(X_val_class)

print('Accuracy:', round(accuracy_score(y_val_class, y_pred_log), 2))
print('Precision:', round(precision_score(y_val_class, y_pred_log), 2))
print('Recall:', round(recall_score(y_val_class, y_pred_log), 2))
print('F1:', round(f1_score(y_val_class, y_pred_log), 2))

# 2. LINEAR REGRESSION 
print('LINEAR REGRESSION')
lreg = LinearRegression()

lreg.fit(X_train_reg,y_train_reg)
y_pred_lreg = lreg.predict(X_val_reg)

print('RMSE:', round(np.sqrt(mean_squared_error(y_val_reg, y_pred_lreg)), 2))
print('MSE:', round(mean_squared_error(y_val_reg, y_pred_lreg), 2))
print('R2:', round(r2_score(y_val_reg, y_pred_lreg), 2))
print('MAE:', round(mean_absolute_error(y_val_reg, y_pred_lreg), 2))


#  PHASE 2 
# --------- ADVANCED MODEL DEVELOPMENT ------

# 1. RANDOM FOREST CLASSIFIER
print('---RANDOM FOREST CLASSIFIER---')

rfc = RandomForestClassifier(random_state=42,n_jobs=-1)
rfc.fit(X_train_class, y_train_class)
y_pred_class = rfc.predict(X_val_class)

print('Accuracy:', round(accuracy_score(y_val_class, y_pred_class), 2))
print('Precision:', round(precision_score(y_val_class, y_pred_class), 2))
print('Recall:', round(recall_score(y_val_class, y_pred_class), 2))
print('F1:', round(f1_score(y_val_class, y_pred_class), 2))


# 2. RANDOM FOREST REGRESSOR 
print('---RANDOM FOREST REGRESSOR---')

rfg = RandomForestRegressor(random_state=42, n_jobs=-1)
rfg.fit(X_train_reg, y_train_reg)
y_pred_reg = rfg.predict(X_val_reg)

print('RMSE:', round(np.sqrt(mean_squared_error(y_val_reg, y_pred_reg)), 2))
print('MSE:', round(mean_squared_error(y_val_reg, y_pred_reg), 2))
print('R2:', round(r2_score(y_val_reg, y_pred_reg), 2))
print('MAE:', round(mean_absolute_error(y_val_reg, y_pred_reg), 2))


# 3. XGBOOST CLASSIFIER 
print('---XGBOOST CLASSIFIER---')

xgboostclass = XGBClassifier(
    n_estimators=200,
    random_state=42,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic'
)
xgboostclass.fit(X_train_class, y_train_class)
y_pred_cxgb = xgboostclass.predict(X_val_class)

print('Accuracy:', round(accuracy_score(y_val_class, y_pred_cxgb), 2))
print('Precision:', round(precision_score(y_val_class, y_pred_cxgb), 2))
print('Recall:', round(recall_score(y_val_class, y_pred_cxgb), 2))
print('F1:', round(f1_score(y_val_class, y_pred_cxgb), 2))


# 4. XGBOOST REGRESSOR
print('---XGBOOST REGRESSOR---')

xgboostreg = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    random_state=42,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror'
)
xgboostreg.fit(X_train_reg, y_train_reg)
y_pred_rxgb = xgboostreg.predict(X_val_reg)

print('RMSE:', round(np.sqrt(mean_squared_error(y_val_reg, y_pred_rxgb)), 2))
print('MSE:', round(mean_squared_error(y_val_reg, y_pred_rxgb), 2))
print('R2:', round(r2_score(y_val_reg, y_pred_rxgb), 2))
print('MAE:', round(mean_absolute_error(y_val_reg, y_pred_rxgb), 2))


# LIGHTGBM CLASSIFIER 
print("----LIGHTGBM CLASSIFIER--- ")
lgbmc = LGBMClassifier(
   objective='binary',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    max_depth=-1,
    random_state=42,
    verbose =-1
)
lgbmc.fit(X_train_class,y_train_class)

y_pred_lgbmc = lgbmc.predict(X_val_class)


print('Accuracy:', round(accuracy_score(y_val_class, y_pred_lgbmc), 2))
print('Precision:', round(precision_score(y_val_class, y_pred_lgbmc), 2))
print('Recall:', round(recall_score(y_val_class, y_pred_lgbmc), 2))
print('F1:', round(f1_score(y_val_class, y_pred_lgbmc), 2))

print("----LIGHTGBM REGRESSOR--- ")

lgbmr = LGBMRegressor(
    num_leaves=31,
    max_depth=-1,
    random_state=42,
    n_estimators=200,
    learning_rate=0.05,
    objective ='regression',
    verbose =-1
)

lgbmr.fit(X_train_reg,y_train_reg)

y_pred_lgbmr = lgbmr.predict(X_val_reg)

print('RMSE:', round(np.sqrt(mean_squared_error(y_val_reg, y_pred_lgbmr)), 2))
print('MSE:', round(mean_squared_error(y_val_reg, y_pred_lgbmr), 2))
print('R2:', round(r2_score(y_val_reg, y_pred_lgbmr), 2))
print('MAE:', round(mean_absolute_error(y_val_reg, y_pred_lgbmr), 2))



