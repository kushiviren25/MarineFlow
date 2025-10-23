import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np 
import warnings 
warnings.filterwarnings('ignore')

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

X_train_reg = train_df.drop(columns=[*leaked_cols,'demurrage_amount_usd'])
y_train_reg = train_df['demurrage_amount_usd']

X_val_reg= validate_df.drop(columns=[*leaked_cols,'demurrage_amount_usd'])
y_val_reg = validate_df['demurrage_amount_usd']

X_test_reg = test_df.drop(columns=[*leaked_cols,'demurrage_amount_usd'])
y_test_reg  = test_df['demurrage_amount_usd']

print('--------HYPERPARAMETER TUNING FOR REGRESSOR MODELS-----------')


print('----RANDOM FOREST REGRESSOR----')
rfg = RandomForestRegressor(
    random_state=42,n_jobs=-1)

params = {
    'n_estimators' : [100,200,300],
    'max_depth' : [5,10,15,20],
    'min_samples_split' : [2,5,10],
    'min_samples_leaf' : [1,2,4],
    'bootstrap' : [True,False]
}

rsearch = RandomizedSearchCV(
    estimator=rfg,
    param_distributions= params,
    n_iter=10,
    n_jobs=-1,
    random_state=42,
    verbose =0,
    cv = 5,
    scoring ='neg_mean_squared_error'
)

rsearch.fit(X_train_reg,y_train_reg)

best_params_rfg = rsearch.best_params_

best_rfg = rsearch.best_estimator_

#  VALIDATION DATA FOR REGRESSION

y_pred_reg = best_rfg.predict(X_val_reg)

print('---VALIDATION DATA RANDOM FOREST REGRESSOR---')
print('RMSE:', round(np.sqrt(mean_squared_error(y_val_reg, y_pred_reg)), 2))
print('MSE:', round(mean_squared_error(y_val_reg, y_pred_reg), 2))
print('R2:', round(r2_score(y_val_reg, y_pred_reg), 2))
print('MAE:', round(mean_absolute_error(y_val_reg, y_pred_reg), 2))

# TEST DATA FOR REGRESSION

y_test_reg_pred = best_rfg.predict(X_test_reg)

print('---TEST DATA FOR RANDOM FOREST REGRESSOR---')
print('RMSE:', round(np.sqrt(mean_squared_error(y_test_reg, y_test_reg_pred)), 2))
print('MSE:', round(mean_squared_error(y_test_reg, y_test_reg_pred), 2))
print('R2:', round(r2_score(y_test_reg, y_test_reg_pred), 2))
print('MAE:', round(mean_absolute_error(y_test_reg, y_test_reg_pred), 2))

print('----XGBOOST REGRESSOR----')

xgbr = XGBRegressor(
    random_state=42,
    n_jobs=-1,
    objective='reg:squarederror',
    eval_metric='rmse'
)

xgbr_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

xgbr_search = RandomizedSearchCV(
    estimator=xgbr,
    param_distributions=xgbr_params,
    n_iter=10,
    n_jobs=-1,
    random_state=42,
    verbose=0,
    cv=5,
    scoring='neg_mean_squared_error'
)

xgbr_search.fit(X_train_reg, y_train_reg)

best_params_xgbr = xgbr_search.best_params_

best_xgbr = xgbr_search.best_estimator_

# VALIDATION REGRESSION DATA 
y_val_reg_pred_xgb = best_xgbr.predict(X_val_reg)

print('---VALIDATION DATA XGBOOST REGRESSOR ---')
print('RMSE:', round(np.sqrt(mean_squared_error(y_val_reg, y_val_reg_pred_xgb)), 2))
print('MSE:', round(mean_squared_error(y_val_reg, y_val_reg_pred_xgb), 2))
print('R2:', round(r2_score(y_val_reg, y_val_reg_pred_xgb), 2))
print('MAE:', round(mean_absolute_error(y_val_reg, y_val_reg_pred_xgb), 2))

# TEST REGRESSION DATA 
y_test_reg_pred_xgb = best_xgbr.predict(X_test_reg)

print('---TEST DATA XGBOOST REGRESSOR---')
print('RMSE:', round(np.sqrt(mean_squared_error(y_test_reg, y_test_reg_pred_xgb)), 2))
print('MSE:', round(mean_squared_error(y_test_reg, y_test_reg_pred_xgb), 2))
print('R2:', round(r2_score(y_test_reg, y_test_reg_pred_xgb), 2))
print('MAE:', round(mean_absolute_error(y_test_reg, y_test_reg_pred_xgb), 2))


print('---- LIGHT GBM REGRESSOR----')


gbmr = LGBMRegressor(
    random_state=42,
    n_jobs=-1,
    objective='regression',
    verbose = -1

)

gbmr_params = {
    'num_leaves' : [15, 31, 63],
    'max_depth' : [3, 5, 7, -1],
    'learning_rate' : [0.01, 0.05, 0.1],
    'n_estimators' : [100, 200, 300],
    'subsample' : [0.7, 0.8, 0.9],
    'colsample_bytree' : [0.7, 0.8, 0.9],
    'random_state' : [42]
}

randomsearch = RandomizedSearchCV(
    estimator=gbmr,
    param_distributions= gbmr_params,
    n_iter=100,
    n_jobs=-1,
    verbose = 0,
    scoring = 'neg_mean_squared_error',
    cv= 5,
    random_state=42
)
 
randomsearch.fit(X_train_reg,y_train_reg)

best_param_gbmr = randomsearch.best_params_

best_gbmr = randomsearch.best_estimator_


#  ----------- VALIDATION DATA FOR LIGHTGBM REGRESSOR---------
print('-------VALIDATION DATA LIGHTGBM REGRESSOR------')
y_reg_val_gbmr = randomsearch.predict(X_val_reg)
rmse = np.sqrt(round(mean_squared_error(y_val_reg,y_reg_val_gbmr)))
print('RMSE:', rmse)
print('MSE: ', round(mean_squared_error(y_val_reg,y_reg_val_gbmr)))
print('R2 score:',round(r2_score(y_val_reg,y_reg_val_gbmr)))
print('MAE:',round(mean_absolute_error(y_val_reg,y_reg_val_gbmr)))

# ------------ TEST DATA FOR LIGHTGBM REGRESSOR ----------
print('------TEST DATA LIGHTGBM REGRESSOR--------')
y_reg_test_gbmr = randomsearch.predict(X_test_reg)

rmse = np.sqrt(round(mean_squared_error(y_test_reg,y_reg_test_gbmr)))
print('RMSE:', rmse)
print('MSE : ', round (mean_squared_error(y_test_reg,y_reg_test_gbmr)))
print('R2 score :',round(r2_score(y_test_reg,y_reg_test_gbmr)))
print('MAE:',round(mean_absolute_error(y_test_reg,y_reg_test_gbmr)))

def regressor_model():
    best_rfg = RandomForestRegressor(**best_params_rfg)
    best_xgbr = XGBRegressor(**best_params_xgbr)
    best_gbmr = LGBMRegressor(**best_param_gbmr)
    return best_rfg,best_xgbr,best_gbmr
