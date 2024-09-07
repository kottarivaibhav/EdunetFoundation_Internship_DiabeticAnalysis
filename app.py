from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model/diabetes_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = [float(request.form['pregnancies']),
                 float(request.form['glucose']),
                 float(request.form['bloodpressure']),
                 float(request.form['skinthickness']),
                 float(request.form['insulin']),
                 float(request.form['bmi']),
                 float(request.form['diabetespedigreefunction']),
                 float(request.form['age'])]
    
    # Convert form data to numpy array for the model
    data = np.array([form_data])
    
    # Make prediction
    prediction = model.predict(data)[0]
    
    # Return result
    result = 'Diabetic' if prediction == 1 else 'Not Diabetic'
    
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
