from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.ensemble import StackingRegressor
from modeldev.TunedRegressor import regressor_model
import pandas as pd 
import numpy as np 

# Load data
train_df = pd.read_csv('marineflow_train.csv')
test_df = pd.read_csv('marineflow_test.csv')

leaked_cols = [
    'overage_hours_chargeable',
    'laytime_vs_allowed',
    'demurrage_amount_usd',
    'demurrage_rate_usd_per_day',
    'demurrage_flag'
]

X_train = train_df.drop(columns=[*leaked_cols, 'demurrage_amount_usd'])
y_train = train_df['demurrage_amount_usd']

X_test = test_df.drop(columns=[*leaked_cols, 'demurrage_amount_usd'])
y_test = test_df['demurrage_amount_usd']


best_rfg,best_xgbr,best_gbmr = regressor_model()

base_estimator = [

    ('rfg',best_rfg),
    ('xgbr',best_xgbr),
    ('gbmr',best_gbmr),
]


meta_model = LinearRegression()

stacking_reg = StackingRegressor(
    estimators = base_estimator,
    final_estimator=meta_model,
    cv= 5 
)

stacking_reg.fit(X_train,y_train)

y_pred = stacking_reg.predict(X_test)

print('----EVALUTION METRICS FOR STACKING REGRESSOR---')
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print('RMSE:',rmse)
print('MSE :',mean_squared_error(y_test,y_pred))
print('MAE :',mean_absolute_error(y_test,y_pred))
print('R2 score :',r2_score(y_test,y_pred))

