import numpy as np
import pandas as pd 
from modeldev.TunedRegressor import regressor_model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from scipy.optimize import minimize
import sys , io

# Data loading :
train_df = pd.read_csv('marineflow_train.csv')
test_df = pd.read_csv('marineflow_test.csv')

# Leaked Columns :
leaked_cols = [ 
    'overage_hours_chargeable',
    'laytime_vs_allowed',
    'demurrage_amount_usd',
    'demurrage_rate_usd_per_day'
]

#  Split to training and testing datasets

X_train = train_df.drop(columns=[*leaked_cols,'demurrage_amount_usd'])
y_train = train_df['demurrage_amount_usd']

X_test = test_df.drop(columns=[*leaked_cols,'demurrage_amount_usd'])
y_test = test_df['demurrage_amount_usd']

silent_stdout = io.StringIO()
original_stdout = sys.stdout
sys.stdout = silent_stdout

# Call the tuned regressor model 
best_rfg, best_xgbr, best_gbmr = regressor_model()

sys.stdout = original_stdout

#  Start with blending 
#  Create separate training and testing data for the 3 models 

y_train_rfg = best_rfg.predict(X_train)
y_train_xgbr = best_xgbr.predict(X_train)
y_train_gbmr = best_gbmr.predict(X_train)


y_test_rfg = best_rfg.predict(X_test)
y_test_xgbr = best_xgbr.predict(X_test)
y_test_gbmr = best_gbmr.predict(X_test)

# Stack the model's predicted values 
predicted_values = np.vstack([y_train_rfg,y_train_xgbr,y_train_gbmr]).T

# Perform mse on these predicted values along with the weights 
def mse_loss(weights):
    blended = np.dot(predicted_values,weights)
    return mean_squared_error(y_train,blended)

constraints = {'type':'eq','fun':lambda w:1 - sum(w)}
bounds = [(0,1)]*3
initial_weights = [1/3,1/3,1/3]

optimised = minimize(mse_loss,initial_weights,constraints=constraints,bounds=bounds)

# Find the optimal weights
optimal_weights = optimised.x
print('Optimal Weights :',optimal_weights)

# Apply these optimal weights to the testing data of the models

y_test_blended = (
    optimal_weights[0] * y_test_rfg +
    optimal_weights[1] * y_test_xgbr +
    optimal_weights[2] * y_test_gbmr
)

# Perform Eval metrics 
print("\n---- EVALUATION METRICS FOR BLENDED REGRESSOR ----")
print('MSE:',mean_squared_error(y_test,y_test_blended))
print('RMSE:',np.sqrt(mean_squared_error(y_test,y_test_blended)))
print('R2:',r2_score(y_test,y_test_blended))



#  Function to import blending regressor 
class BlendingRegressor:
    def __init__(self,models,weights):
        self.models = models
        self.weights = weights

    def predict(self,X):
        pred = [model.predict(X) for model in self.models]
        blended = np.dot(np.vstack(pred).T,self.weights)
        return blended 
    
blended_reg = BlendingRegressor(
    weights= optimal_weights,
    models = [best_rfg, best_xgbr, best_gbmr]
)   
        
