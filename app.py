from flask import Flask, request, render_template
import joblib
import numpy as np

# # Load the model and scaler
model = joblib.load('PriceModel.pkl')
scaler = joblib.load('PriceScaler.pkl')

app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        Brand = int(request.form['Brand'])
        Storage = int(request.form['Storage'])
        RAM = int(request.form['RAM'])
        ScreenSize = float(request.form['Screen Size (inches)'])
        BatteryCapacity = float(request.form['Battery Capacity (mAh)'])

        
        # Combine all features into a single NumPy array
        features = np.array([[Brand, Storage, RAM, ScreenSize, BatteryCapacity]])

        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        price = round(model.predict(features_scaled)[0]*87.2,2)
        
        return render_template('index.html', prediction_text=f'Predicted Price: ₹ {price}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)