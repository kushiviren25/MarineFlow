import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
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
   verbose = 2,
   cv = 5
)

# fit the training data 
random_search.fit(X_train_class,y_train_class)

# best parameters 
print(random_search.best_params_)

# call the best estimator 
best_rfc = random_search.best_estimator_

# evaluate the validation data 
y_pred_class = random_search.predict(X_val_class)

# Eval metrics for validation
print('VALIDATION DATA CLASSIFICATION')

print('Accuracy:',accuracy_score(y_val_class,y_pred_class))
print('Precision:',precision_score(y_val_class,y_pred_class))
print('Recall:',recall_score(y_val_class,y_pred_class))
print('F1:',f1_score(y_val_class,y_pred_class))

print('TEST DATA CLASSIFICATION')

y_test_class_pred = best_rfc.predict(X_test_class)

print('Accuracy:',accuracy_score(y_test_class,y_test_class_pred ))
print('Precision:',precision_score(y_test_class,y_test_class_pred ))
print('Recall:',recall_score(y_test_class,y_test_class_pred ))
print('F1:',f1_score(y_test_class,y_test_class_pred ))


# 2. RANDOM FOREST REGRESSOR 

rfg = RandomForestRegressor(random_state=42,n_jobs=-1)

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
    verbose =2,
    cv = 5,
    scoring ='neg_mean_squared_error'
)

rsearch.fit(X_train_reg,y_train_reg)

print('best params',rsearch.best_params_)

best_rfg = rsearch.best_estimator_

print('VALIDATION DATA FOR REGRESSION')
y_pred_reg = best_rfg.predict(X_val_reg)

rmse = np.sqrt(mean_squared_error(y_val_reg,y_pred_reg))

print('RMSE:',rmse)
print('MSE:',mean_squared_error(y_val_reg,y_pred_reg))
print('R2:',r2_score(y_val_reg,y_pred_reg))
print('MAE:',mean_absolute_error(y_val_reg,y_pred_reg))

print('TEST DATA FOR REGRESSION')

y_test_reg_pred = best_rfg.predict(X_test_reg)

rmse = np.sqrt(mean_squared_error(y_test_reg,y_test_reg_pred))

print('RMSE:',rmse)
print('MSE:',mean_squared_error(y_test_reg,y_test_reg_pred))
print('R2:',r2_score(y_test_reg,y_test_reg_pred))
print('MAE:',mean_absolute_error(y_test_reg,y_test_reg_pred))




