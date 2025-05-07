import streamlit as st 
import numpy as np 
import joblib 

# Load the model
model = joblib.load("rf_model.pkl")

# App's title
st.title("Laptop Price Prediction App")

st.divider()

# App's functionality description 
st.write("With this app after filling the above fields and using the calculation button you can get a price estimation for the laptop you want.")

st.divider()

# Take the predictors values
processor_speed = st.number_input("Enter laptop's processor speed: ", value=2.5, step=0.5)
ram_size = st.number_input("Enter laptop's RAM size: ", value=16,
step=4)
storage_capacity = st.number_input("Enter laptop's storage capacity: ", value=512, step=256)

# Store the predictor values in an array 
X = [processor_speed, ram_size, storage_capacity]

st.divider()

# Initialize the prediction button 
prediction = st.button("Click the button to get the laptop's price prediction")

st.divider()

if prediction:
    st.balloons()
    X1 = np.array(X)
    prediction = model.predict([X1])[0]
    st.write(f"Price Estimation For The Laptop With The Given Characteristics  Is: {prediction:,2f}")
else:
    st.write("Please press the button to get the prediction price for the laptop you want")



