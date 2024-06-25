# libraries- flask, scikit-learn, pandas, pickle-mixin, flask-cors, babel
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
from babel.numbers import format_currency

app = Flask(__name__)
data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open("RidgeModel(new).pkl", "rb"))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))
    print(location, bhk, bath, sqft)

    # Create input DataFrame with the correct column names
    input_df = pd.DataFrame([[location, sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'bhk'])

    # Use the pipeline to transform the input and predict
    prediction = pipe.predict(input_df)[0] * 1e5

    # Format the prediction with commas in the Indian numbering system
    formatted_prediction = format_currency(prediction, 'INR', locale='en_IN')

    return str(formatted_prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5001)