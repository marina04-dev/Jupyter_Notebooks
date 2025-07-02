import streamlit as st
import pandas as pd
import joblib

model = joblib.load('fraud_detection_model.pkl')

st.title('Fraud Detection Prediction Application')

st.markdown('Please Enter The Transaction Details And Use The Predict Button To Get The Prediction Output')

st.divider()

transaction_type = st.selectbox('Transaction Type', ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT'])
amount = st.number_input('Amount', min_value = 0.0, value = 1000.0)
oldBalanceOrg = st.number_input('Old Balance (Sender)', min_value = 0.0, value = 1000.0)
newBalanceOrg = st.number_input('New Balance (Sender)', min_value = 0.0, value = 1000.0)
oldBalanceDest = st.number_input('Old Balance (Receiver)', min_value = 0.0, value = 1000.0)
newBalanceDest = st.number_input('New Balance (Receiver)', min_value = 0.0, value = 1000.0)

if st.button('Predict'):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldBalanceOrg,
        "newbalanceOrig": newBalanceOrg,
        "oldbalanceDest": oldBalanceDest,
        "newbalanceDest": newBalanceDest
    }])
    prediction = model.predict(input_data)[0]
    st.subheader(f"Prediction : {int(prediction)}")

    if prediction == 1:
        st.error('This Transaction Can Lead To Fraud!')
    else:
        st.success('This Transaction Seems To Be Not Fraud!')
