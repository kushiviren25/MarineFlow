from flask import Flask,request,jsonify
from flask_cors import CORS
import pandas as pd 
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from riskframework.riskframe import RiskFramework
import traceback
import logging

# Initialise Flask 
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Call Framework 
framework = RiskFramework(use_log_transform=False,min_amount_threshold=100)

# check if Flask is working 
@app.route("/")
def home():
    "Checking if Flask works"
    return jsonify({'message':"Flask is now live!"})

@app.route("/predict",methods=['POST'])
def predict():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error':'Couldnt find the file'}),400
        
        df = pd.read_csv(file)
        
        demurrage_amount,demurrage_flag,class_prob = framework.predict_risk(df)

        demurrage_flag = pd.Series(demurrage_flag).astype(int).tolist()
        class_prob= pd.Series(class_prob).astype(float).tolist()
        demurrage_amount = pd.Series(demurrage_amount).astype(float).tolist()

        results = pd.DataFrame({
        "Demurrage Flags" : demurrage_flag,
        "Demurrage Probability" : class_prob,
        "Demurrage Amount" : demurrage_amount
        })
        summary = {
            "Total Cases" : len(df),
            "Total Demurrage Flags" : int(sum(demurrage_flag)),
            "Max Demurrage Amount" : float(max(demurrage_amount))
            }
        return jsonify({
            "Results" : results.to_dict(orient="records"),
            "Summary" : summary
            }),200
    except Exception as e:
        logging.error(traceback.format_exc())
        return jsonify({"error":str(e)}) ,500
    
if __name__ == "__main__":
    print('Starting Flask Server')
    print('Framework loaded',framework)
    app.run(host="0.0.0.0",port = 5000,debug=True,use_reloader=False) 
    


