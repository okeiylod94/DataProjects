# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.classification import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("final_best_api")

# Create input/output pydantic models
input_model = create_model("final_best_api_input", **{'Transaction_Amount': 4659.259765625, 'Transaction_Type': 'POS Payment', 'Time_of_Transaction': 21.0, 'Device_Used': 'Mobile', 'Location': 'Miami', 'Previous_Fraudulent_Transactions': 3, 'Account_Age': 48, 'Number_of_Transactions_Last_24H': 1, 'Payment_Method': 'UPI'})
output_model = create_model("final_best_api_output", prediction=0)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
