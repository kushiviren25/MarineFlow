import os 
import sys
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeldev.stackingclassifier import stacking_classifier
from modeldev.blendedregressor import blended_reg

# Logging setup
logging.basicConfig(
    filename='risk_assessment.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RiskFramework:
    def __init__(self, classifier=stacking_classifier, regressor=blended_reg,
                 use_log_transform=False, min_amount_threshold=100):
        self.classifier = classifier
        self.regressor = regressor
        self.use_log_transform = use_log_transform
        self.min_amount_threshold = min_amount_threshold
        logging.info(f"Framework initialized | LogTransform: {use_log_transform} | MinThreshold: ${min_amount_threshold}")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [
            'overage_hours_chargeable',
            'laytime_vs_allowed',
            'demurrage_amount_usd',
            'demurrage_rate_usd_per_day'
        ]
        return df.drop(columns=cols_to_drop, errors='ignore').fillna(0)

    def predict_risk(self, df: pd.DataFrame):
        X = self.preprocess(df)
        if X.empty:
            raise ValueError("Empty dataframe provided for prediction.")

        X_class = X.drop(columns=['demurrage_flag'], errors='ignore')

        class_prob = self.classifier.predict_proba(X_class)[:, 1]

        # Classification of Demurrage using probability 
        threshold  = 0.5
        demurrage_flag = (class_prob >threshold).astype(int)

        # Prediction of Demurrage Amount
        demurrage_amount = self.regressor.predict(X)

    
        if self.use_log_transform:
            # for skewed data 
            demurrage_amount = np.expm1(demurrage_amount)  
        else:
            # for negative values
           demurrage_amount= np.maximum(0, demurrage_amount)
        

        # Apply threshold synchronization for classofication and regression 

        # if flag = 1 , print amount if not set amount  = 0
        demurrage_amount = np.where(demurrage_flag == 1,demurrage_amount,0)

        # if demurrage amount lesser  than min threshold value , set amount  =0
        demurrage_amount = np.where(demurrage_amount > self.min_amount_threshold,demurrage_amount,0)

        # if demurrage amount greater than 0 , flag it as 1 
        demurrage_flag = np.where(demurrage_amount > 0, 1 , 0)
     
        class_prob = np.round(class_prob,4)
        demurrage_amount = np.round(demurrage_amount,2)
        
        return demurrage_amount,demurrage_flag,class_prob


# ----- MAIN EXECUTION -----
if __name__ == "__main__":
    df = pd.read_csv('marineflow_test.csv')
    framework = RiskFramework(use_log_transform=False, min_amount_threshold=100)
    
    demurrage_amount,demurrage_flag,class_prob = framework.predict_risk(df)

    print("-------- RISK PREDICTION SUMMARY --------")
    print(f"Total Cases: {len(demurrage_amount)}")
    print(f"Demurrage Flags: {np.sum(demurrage_flag)}")
    print(f"Average Predicted Amount (Demurrage): ${demurrage_amount[demurrage_flag==1].mean():.2f}")
    print(f"Max Predicted Amount: ${demurrage_amount.max():.2f}")

    print("\n--- Sample Predictions ---")
    for i in range(min(5, len(demurrage_flag))):
        status = "DEMURRAGE" if demurrage_flag[i] else "NO DEMURRAGE"
        print(f"Case {i+1}: {status} | Prob: {class_prob[i]:.2%} | Amount: ${demurrage_amount[i]:.2f}")
