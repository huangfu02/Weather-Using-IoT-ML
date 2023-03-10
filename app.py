import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load the scaler object
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the model
model = load_model('my_model.h5')

# Load the dataset
df = pd.read_csv('clean_data.csv')

# Create Flask app
app = Flask(__name__)


# Define routes
@app.route('/')
def home():
    return render_template('index.html', df=df)


@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    input_data = []
    for feature in df.columns[:-1]:  # exclude the target column "rain"
        value = float(request.form.get(feature))
        input_data.append(value)

    # Scale the user input
    input_data = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(scaled_input)
    prediction = np.round(prediction).astype(int)[0][0]

    # Format the result as a string
    if prediction == 1:
        result = 'It will rain.'
    else:
        result = 'It will not rain.'

    # Render the results on the web page
    return render_template('index.html', prediction_text=result, df=df)


if __name__ == '__main__':
    app.run(debug=True)
