from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model once when the app starts
model = joblib.load('model/heart_disease_model.pkl')

# Feature order MUST match how the model was trained
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [float(request.form[feature]) for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        result = "High risk of Heart Disease" if prediction == 1 else "Low risk of Heart Disease"
        confidence = round(probability * 100, 2) if prediction == 1 else round((1 - probability) * 100, 2)

        return render_template('index.html', result=result, confidence=confidence)

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)