from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

# Function to fetch data and train the model
def train_model():
    # Fetch data from the API
    url = 'https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol=USD&to_symbol=PKR&apikey=WYXNGVUVCHVMD6W1'
    r = requests.get(url)
    data = r.json()
    
    if "Time Series FX (Monthly)" in data:
        fx_data = data["Time Series FX (Monthly)"]
        
        # Convert JSON data to DataFrame
        df = pd.DataFrame(fx_data).T
        df.columns = ['Open', 'High', 'Low', 'Close']
        df = df.apply(pd.to_numeric)
        df.index = pd.to_datetime(df.index)
        
        # Independent and dependent variables
        X = df[['Open', 'High', 'Low']]  # Features
        Y = df['Close']  # Target variable
        
        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=20)
        
        # Train the linear regression model
        lr = LinearRegression()
        lr.fit(X_train, Y_train)
        
        # Make predictions
        Y_pred = lr.predict(X_test)
        
        # Calculate the R² score
        r2 = r2_score(Y_test, Y_pred)
        
        return lr, r2, X_test, Y_test
    
    return None, None, None, None

# Route for the homepage (input form)
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get values from the input form
        open_value = float(request.form['open'])
        high_value = float(request.form['high'])
        low_value = float(request.form['low'])
        
        # Reshape input to match the model's expected format
        input_data = np.array([open_value, high_value, low_value]).reshape(1, -1)
        
        # Train the model and get predictions
        model, r2, X_test, Y_test = train_model()
        
        if model is not None:
            # Predict using the model
            prediction = model.predict(input_data)
            
            return render_template('result.html', prediction=prediction[0], r2=r2)
        else:
            return "Failed to fetch data or train the model."

if __name__ == "__main__":
    app.run(debug=True)
