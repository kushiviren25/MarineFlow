from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score,r2_score,f1_score,precision_score,recall_score,mean_squared_error
import joblib
import pandas as pd
import numpy as np

train_df = pd.read_csv('marineflow_train.csv')
validate_df = pd.read_csv('marineflow_validation.csv')
test_df = pd.read_csv('marineflow_test.csv') 

# Features 
X_train = train_df.drop(columns=['demurrage_flag','demurrage_amount_usd'])
X_validate = validate_df.drop(columns=['demurrage_flag','demurrage_amount_usd'])
X_test = test_df.drop(columns=['demurrage_flag','demurrage_amount_usd'])

# Target for classsification 
y_train_class = train_df['demurrage_flag']
y_validate_class = validate_df['demurrage_flag']
y_test_class = test_df['demurrage_flag']

# Target for Regression 
y_train_reg = train_df['demurrage_amount_usd']
y_validate_reg = validate_df['demurrage_amount_usd']
y_test_reg = test_df['demurrage_amount_usd']


# Logisitic Regression 
logreg = LogisticRegression()

logreg.fit(X_train,y_train_class)

y_pred_class = logreg.predict(X_validate)

print("----EVALUATION METRICS FOR CLASSIFICATION----")

print("Validation Accuracy Score :", accuracy_score(y_validate_class,y_pred_class))
print("Validation Precision Sccore :", precision_score(y_validate_class,y_pred_class))
print("Validation f1 Score :",f1_score(y_validate_class,y_pred_class))
print("Validation Recall score :", recall_score(y_validate_class,y_pred_class))



# Linear Regression
lreg = LinearRegression()

lreg.fit(X_train,y_train_reg)

y_pred_reg = lreg.predict(X_validate)

print("----EVALUATION METRICS FOR REGRESSION----")

print("Validated MSE:",mean_squared_error(y_validate_reg,y_pred_reg))
rmse = np.sqrt(mean_squared_error(y_validate_reg,y_pred_reg))
print("Validated RMSE:",rmse)
print("Validated R2:",r2_score(y_validate_reg,y_pred_reg))
