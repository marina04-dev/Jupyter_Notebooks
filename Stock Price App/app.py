import yfinance as yf
import streamlit as st
import pandas as pd
import time
from yfinance.exceptions import YFRateLimitError

st.write("""
# Simple Stock Price App
         
Shown are the stock closing price and volume of Google!

""")

# https://towardsdatascience.com/how-to-get-data-using-python-c0de1df17e75

# Creates a variable containing Google's stock ticker symbol
tickerSymbol = 'GOOGL'

@st.cache_data # Streamlit decorator that caches the function's results. This means if you run the app again with the same ticker, it uses saved data instead of making a new API call
def load_data(ticker_symbol):
    """Load stock data with error handling and caching"""
    max_retries = 3 # Sets how many times we'll try if the API call fails
    
    for attempt in range(max_retries):
        try:
            # get data on this ticker
            tickerData = yf.Ticker(ticker_symbol) # Creates a yfinance Ticker object for the stock symbol
            
            # get the historical prices for this ticker
            tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
            
            return tickerDf
            
        except YFRateLimitError: # Catches the specific rate limit error
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # Wait 30, 60, 90 seconds
                # Shows a yellow warning message on the web page
                st.warning(f"Rate limited. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else: # This runs if we've used all retries and still failed
                st.error("Failed to fetch data after multiple attempts. Please try again later.") # Shows a red error message on the web page
                return None # Returns nothing, indicating failure
        except Exception as e: # Catches any other type of 
            # str(e): Converts the error to a readable string. # Shows the actual error message to help with debugging
            st.error(f"Error fetching data: {str(e)}")
            return None

# Load the data
with st.spinner('Loading stock data...'): #  Shows a spinning loading indicator
    tickerDf = load_data(tickerSymbol)

if tickerDf is not None and not tickerDf.empty:
    # Display some basic info
    st.subheader(f"Stock Data for {tickerSymbol}")
    st.write(f"Data from {tickerDf.index[0].strftime('%Y-%m-%d')} to {tickerDf.index[-1].strftime('%Y-%m-%d')}")
    
    # Display the charts
    st.subheader("Closing Price")
    st.line_chart(tickerDf.Close)
    
    st.subheader("Volume")
    st.line_chart(tickerDf.Volume)
    
    # Optional: Show raw data
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.dataframe(tickerDf)
        
    # Optional: Show basic statistics
    if st.checkbox("Show statistics"):
        st.subheader("Basic Statistics")
        st.write(tickerDf.describe())
        
else: # This runs if tickerDf is None or empty
    st.error("Unable to load stock data. Please try again later.")
