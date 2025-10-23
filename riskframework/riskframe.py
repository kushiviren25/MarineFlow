from modeldev import stackingclassifier
from modeldev import stackingregressor
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
    def __init__(self,classifier = stackingclassifier,regressor=stackingregressor):
        self.classifier = classifier
        self.regressor = regressor 

    # preprocess the data before predicting 
    def preprocess(self,df:pd.DataFrame) -> pd.DataFrame:
        leaked_cols = ['overage_hours_chargeable','laytime_vs_allowed','demurrage_amount_usd','demurrage_rate_usd_per_day']

        X = df.drop(columns=leaked_cols,errors='ignore')
        X = X.fillna(0)
        return X
    
    # predicting the risk 
    def predict_risk(self, df: pd.Dataframe):
        
        X = self.preprocess(df)

        # if x is empty
        if X.empty:
            raise ValueError('Input dataframe is empty')
        
        classification = self.classifier.predict(X)
        class_probability = self.classifier.predict_proba(X)[:,1]
        regression = self.regression.predict(X)

        logging.info(f"Input: {X.to_dict(orient='records')}")
        logging.info(f"Predicted class :{classification.tolist()},Probability:{class_probability.tolist()},Risk Score:{regression.tolist()}")

        return classification,class_probability,regression
        


