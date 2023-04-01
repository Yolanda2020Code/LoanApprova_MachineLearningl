from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and scaler
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    gender = features[0]
    married = features[1]
    dependents = features[2]
    education = features[3]
    self_employed = features[4]
    applicant_income = int(features[5])
    coapplicant_income = int(features[6])
    loan_amount = int(features[7])
    loan_amount_term = int(features[8])
    credit_history = int(features[9])
    property_area = features[10]

    # Preprocess the input data
    new_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })
    new_data = pd.get_dummies(new_data)
    new_data = scaler.transform(new_data)

    # Make loan approval prediction
    prediction = model.predict(new_data)[0]
    if prediction == 1:
        result = 'Loan Approved'
    else:
        result = 'Loan Not Approved'

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)