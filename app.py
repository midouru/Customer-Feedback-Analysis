from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    rating = float(request.form['rating'])
    feedback = float(request.form['feedback'])

    input_Data = np.array([[age, rating, feedback]])

    prediction = model.predict(input_Data)[0]
    if prediction == 1:
        result = "✅recommended."
        result_class = "positive"
    else:
        result = " ❌ Not recommended."
        result_class = "negative"
    return render_template('index.html', prediction_text=result, result_class=result_class)

if __name__ == "__main__":
    app.run(debug=True)

