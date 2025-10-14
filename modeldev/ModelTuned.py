import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
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


# -------- HYPERPARAMETER TUNING ---------------

# 1. RANDOM FOREST CLASSIFIER

rfc = RandomForestClassifier(random_state=42,n_jobs=-1)

parameters = {
    'n_estimators':[100,200,300],
    'max_depth': [5,10,15,20],
    'min_samples_split':[2,5,10],
    'min_samples_leaf' : [1,2,4],
    'bootstrap' : [True,False] 
}

random_search = RandomizedSearchCV(
   
   estimator=rfc,
   param_distributions= parameters,
   n_iter=10,
   n_jobs=-1,
   random_state=42,
   scoring='f1_macro',
   verbose = 0,
   cv = 5
)

# fit the training data 
random_search.fit(X_train_class,y_train_class)

# best parameters 
best_params_rfc = random_search.best_params_

# best estimator 
best_rfc = random_search.best_estimator_


y_pred_class = random_search.predict(X_val_class)

print('----VALIDATION DATA CLASSIFICATION----')
print('Accuracy:', round(accuracy_score(y_val_class, y_pred_class), 2))
print('Precision:', round(precision_score(y_val_class, y_pred_class), 2))
print('Recall:', round(recall_score(y_val_class, y_pred_class), 2))
print('F1:', round(f1_score(y_val_class, y_pred_class), 2))

y_test_class_pred = best_rfc.predict(X_test_class)

print('----TEST DATA CLASSIFICATION----')
print('Accuracy:', round(accuracy_score(y_test_class, y_test_class_pred), 2))
print('Precision:', round(precision_score(y_test_class, y_test_class_pred), 2))
print('Recall:', round(recall_score(y_test_class, y_test_class_pred), 2))
print('F1:', round(f1_score(y_test_class, y_test_class_pred), 2))

# 2. RANDOM FOREST REGRESSOR 

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

print('best params',rsearch.best_params_)

best_rfg = rsearch.best_estimator_

#  VALIDATION DATA FOR REGRESSION

y_pred_reg = best_rfg.predict(X_val_reg)

print('---VALIDATION DATA FOR REGRESSION---')
print('RMSE:', round(np.sqrt(mean_squared_error(y_val_reg, y_pred_reg)), 2))
print('MSE:', round(mean_squared_error(y_val_reg, y_pred_reg), 2))
print('R2:', round(r2_score(y_val_reg, y_pred_reg), 2))
print('MAE:', round(mean_absolute_error(y_val_reg, y_pred_reg), 2))

# TEST DATA FOR REGRESSION

y_test_reg_pred = best_rfg.predict(X_test_reg)

print('---TEST DATA FOR REGRESSION---')
print('RMSE:', round(np.sqrt(mean_squared_error(y_test_reg, y_test_reg_pred)), 2))
print('MSE:', round(mean_squared_error(y_test_reg, y_test_reg_pred), 2))
print('R2:', round(r2_score(y_test_reg, y_test_reg_pred), 2))
print('MAE:', round(mean_absolute_error(y_test_reg, y_test_reg_pred), 2))

# 3. XGBOOST CLASSIFIER

print('XGBOOST CLASSIFIER')
xgbc = XGBClassifier(
    random_state=42,
    n_jobs=-1,
    objective='binary:logistic',
    eval_metric='logloss'
)

xgbc_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

xgbc_search = RandomizedSearchCV(
    estimator=xgbc,
    param_distributions=xgbc_params,
    n_iter=10,
    n_jobs=-1,
    random_state=42,
    scoring='f1_macro',
    verbose=2,
    cv=5
)

xgbc_search.fit(X_train_class, y_train_class)

print('Best Parameters for XGBoost Classifier:', xgbc_search.best_params_)

best_xgbc = xgbc_search.best_estimator_
best_params_xgbc = xgbc_search.best_params_

# VALIDATION CLASSIFICATION DATA 
y_val_class_pred_xgb = best_xgbc.predict(X_val_class)

print('----VALIDATION DATA CLASSIFICATION (XGBOOST)----')
print('Accuracy:', round(accuracy_score(y_val_class, y_val_class_pred_xgb), 2))
print('Precision:', round(precision_score(y_val_class, y_val_class_pred_xgb), 2))
print('Recall:', round(recall_score(y_val_class, y_val_class_pred_xgb), 2))
print('F1:', round(f1_score(y_val_class, y_val_class_pred_xgb), 2))

# TEST CLASSIFICATION DATA 
y_test_class_pred_xgb = best_xgbc.predict(X_test_class)

print('----TEST DATA CLASSIFICATION (XGBOOST)----')
print('Accuracy:', round(accuracy_score(y_test_class, y_test_class_pred_xgb), 2))
print('Precision:', round(precision_score(y_test_class, y_test_class_pred_xgb), 2))
print('Recall:', round(recall_score(y_test_class, y_test_class_pred_xgb), 2))
print('F1:', round(f1_score(y_test_class, y_test_class_pred_xgb), 2))




#  XGBOOST REGRESSOR
print('XGBOOST REGRESSOR')

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
    verbose=2,
    cv=5,
    scoring='neg_mean_squared_error'
)

xgbr_search.fit(X_train_reg, y_train_reg)

print('Best Parameters for XGBoost Regressor:', xgbr_search.best_params_)

best_xgbr = xgbr_search.best_estimator_

# VALIDATION REGRESSION DATA 
y_val_reg_pred_xgb = best_xgbr.predict(X_val_reg)

print('---VALIDATION DATA FOR REGRESSION (XGBOOST)---')
print('RMSE:', round(np.sqrt(mean_squared_error(y_val_reg, y_val_reg_pred_xgb)), 2))
print('MSE:', round(mean_squared_error(y_val_reg, y_val_reg_pred_xgb), 2))
print('R2:', round(r2_score(y_val_reg, y_val_reg_pred_xgb), 2))
print('MAE:', round(mean_absolute_error(y_val_reg, y_val_reg_pred_xgb), 2))

# TEST REGRESSION DATA 
y_test_reg_pred_xgb = best_xgbr.predict(X_test_reg)

print('---TEST DATA FOR REGRESSION (XGBOOST)---')
print('RMSE:', round(np.sqrt(mean_squared_error(y_test_reg, y_test_reg_pred_xgb)), 2))
print('MSE:', round(mean_squared_error(y_test_reg, y_test_reg_pred_xgb), 2))
print('R2:', round(r2_score(y_test_reg, y_test_reg_pred_xgb), 2))
print('MAE:', round(mean_absolute_error(y_test_reg, y_test_reg_pred_xgb), 2))

# HYPERPARAMETER TUNING FOR LIGHTGBM 

# LIGHT GBM CLASSIFIER 
lightgbmc = LGBMClassifier(
    n_jobs=-1,
    objective='binary',
    random_state=42
)

gbmc_params = {
    'num_leaves' : [15, 31, 63],
    'max_depth' : [3, 5, 7, -1],
    'learning_rate' : [0.01, 0.05, 0.1],
    'n_estimators' : [100, 200, 300],
    'subsample' : [0.7, 0.8, 0.9],
    'colsample_bytree' : [0.7, 0.8, 0.9],
    'random_state' : [42]
}


ran_search = RandomizedSearchCV(
    estimator= lightgbmc,
    param_distributions= gbmc_params,
    n_iter=10,
    n_jobs=-1,
    random_state=42,
    cv = 5 ,
    verbose = 2,
    scoring = 'f1_macro'
)


ran_search.fit(X_train_class,y_train_class)

best_gbmc_params = ran_search.best_params_

best_gbmc = ran_search.best_estimator_

y_val_class_pred_gbmc = ran_search.predict(X_val_class)

print('----VALIDATION DATA CLASSIFICATION (LIGHT GBM)----')
print('Accuracy:', round(accuracy_score(y_val_class, y_val_class_pred_gbmc), 2))
print('Precision:', round(precision_score(y_val_class, y_val_class_pred_gbmc), 2))
print('Recall:', round(recall_score(y_val_class, y_val_class_pred_gbmc), 2))
print('F1:', round(f1_score(y_val_class, y_val_class_pred_gbmc), 2))


y_test_class_pred_gbmc = ran_search.predict(X_test_class)

print('---- TESTING DATA CLASSIFICATION (LIGHT GBM)----')
print('Accuracy:', round(accuracy_score(y_test_class,y_test_class_pred_gbmc), 2))
print('Precision:', round(precision_score(y_test_class,y_test_class_pred_gbmc), 2))
print('Recall:', round(recall_score(y_test_class,y_test_class_pred_gbmc), 2))
print('F1:', round(f1_score(y_test_class, y_test_class_pred_gbmc), 2))





# # # FOR STACKING CLASSIFIER 
def get_class_model():
    best_rfc = RandomForestClassifier(**best_params_rfc)
    best_xgbc = XGBClassifier(**best_params_xgbc)
    best_gbmc = LGBMClassifier(**best_gbmc_params)
    return best_rfc,best_xgbc,best_gbmc