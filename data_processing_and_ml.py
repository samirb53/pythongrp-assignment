# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import requests
import os
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class SimFinAPI:
    def __init__(self, api_key):
        self.base_url = "https://simfin.com/api/v2/"
        self.headers = {"Authorization": f"api-key {api_key}"}
    
    def get_company_id(self, ticker):
        url = f"{self.base_url}companies/list"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            companies = response.json()
            for company in companies:
                if company["ticker"] == ticker:
                    return company["simId"]
        print(f"Error: Company ID not found for ticker {ticker}.")
        return None
    
    def get_share_prices(self, ticker, start_date, end_date):
        company_id = self.get_company_id(ticker)
        if not company_id:
            return None
        
        url = f"{self.base_url}companies/id/{company_id}/shares/prices"
        params = {"start": start_date.strftime("%Y-%m-%d"), "end": end_date.strftime("%Y-%m-%d")}
        response = requests.get(url, headers=self.headers, params=params)
        
        print(f"API Response Status: {response.status_code}")
        print(f"API Response Text: {response.text}")
        
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            return None

# Step 3: Data Preprocessing & Feature Engineering
def preprocess_data(data):
    data["Date"] = pd.to_datetime(data["Date"])
    data.sort_values("Date", inplace=True)
    data["Price_Change"] = data["Close"].diff()
    data["Target"] = np.where(data["Price_Change"] > 0, 1, 0)
    data.dropna(inplace=True)
    return data

# Step 4: Train Machine Learning Model
def train_model(data):
    features = ["Open", "High", "Low", "Close", "Volume"]
    X = data[features]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return model, accuracy

# Step 5: Streamlit Web Application
def web_app():
    st.title("Automated Trading System")
    st.sidebar.header("Select Stock")
    ticker = st.sidebar.text_input("Enter Stock Ticker:", "AAPL")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
    
    if st.sidebar.button("Fetch Data"):
        api = SimFinAPI(api_key="e1c75cc5-3bca-4b0c-b847-6447bd4ed901")
        data = api.get_share_prices(ticker, start_date, end_date)
        if data is not None and not data.empty:
            st.write("Stock Data:", data.tail())
            processed_data = preprocess_data(data)
            
            if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
                model = joblib.load("model.pkl")
                scaler = joblib.load("scaler.pkl")
            else:
                model, accuracy = train_model(processed_data)
                st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
                
            latest_data = processed_data.iloc[-1][["Open", "High", "Low", "Close", "Volume"]].values.reshape(1, -1)
            latest_data = scaler.transform(latest_data)
            prediction = model.predict(latest_data)[0]
            st.write("Prediction for Next Day:", "Rise" if prediction == 1 else "Fall")
        else:
            st.write("Error fetching data. Check API Key and ticker.")

if __name__ == "__main__":
    web_app()