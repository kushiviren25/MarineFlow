import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeldev.stackingclassifier import stacking_classifier
from modeldev.stackingregressor import stacking_reg
import logging
import pandas as pd 


# Logging setup 

logging.basicConfig(
    filename ='risk_assesment.log',
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s '
)

# Class for Risk Framework
class RiskFramework:
    # setup initializer 
    def __init__(self,classifier = stacking_classifier,regressor=stacking_reg):
        self.classifier = classifier
        self.regressor = regressor 

    # preprocess the data before predicting 
    def preprocess(self,df:pd.DataFrame) -> pd.DataFrame:
        leaked_cols = ['overage_hours_chargeable','laytime_vs_allowed','demurrage_amount_usd','demurrage_rate_usd_per_day','demurrage_flag']

        X = df.drop(columns=leaked_cols,errors='ignore')
        X = X.fillna(0)
        return X
    
    # predicting the risk 
    def predict_risk(self, df: pd.DataFrame):
        
        X = self.preprocess(df)

        # if x is empty
        if X.empty:
            raise ValueError('Input dataframe is empty')
        
        classification = self.classifier.predict(X)
        class_probability = self.classifier.predict_proba(X)[:,1]
        regression = self.regressor.predict(X)

        logging.info(f"Input: {X.to_dict(orient='records')}")
        logging.info(f"Predicted class :{classification.tolist()},Probability:{class_probability.tolist()},Risk Score:{regression.tolist()}")

        return classification,class_probability,regression
        

# Main function for Predicting the Risk 

if __name__ == "__main__" :

    # load the testing csv 
    df = pd.read_csv('marineflow_test.csv')

    #  initialize the risk framework 
    framework = RiskFramework()

    # call the variables 
    classification,class_probability,regression = framework.predict_risk(df)

    # RISK PREDICTION 
    print('--------RISK PREDICTION--------')
    print('Predicted Demurrage Flag :',classification[:5])
    print('Predicted Demurrage Probability :',class_probability[:5])
    print('Predicted Demurrage Amount:',regression[:5])

