from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model (created in Google Colab and downloaded)
model = joblib.load('diabetes_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        input_data = [
            float(request.form['gender']),   # 0 for Male, 1 for Female
            float(request.form['age']),
            float(request.form['urea']),
            float(request.form['cr']),
            float(request.form['hba1c']),
            float(request.form['chol']),
            float(request.form['tg']),
            float(request.form['hdl']),
            float(request.form['ldl']),
            float(request.form['vldl']),
            float(request.form['bmi'])
        ]

        # Convert to numpy array and predict
        input_array = np.array([input_data])
        prediction = model.predict(input_array)[0]

        result = "Diabetic" if prediction == 1 else "Not Diabetic"
    except Exception as e:
        result = f"Prediction error: {str(e)}"

    return render_template('index.html', prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True)
