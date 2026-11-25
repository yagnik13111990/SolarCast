import os
import pickle
import joblib
import numpy as np
import json
from flask import Flask, render_template, request

from Utils.preprocess import preprocess_input

app = Flask(__name__)

#load model 
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model', 'best_model.pkl')

try:
    model_data = joblib.load(MODEL_PATH)
except Exception:
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        model_data = None 
        print(f"CRITICAL ERROR: Could not load model. {e}")

def polynomial_feature_inf(X, degree = 2):
    n_samples, n_features = X.shape
    feat = [np.ones(n_samples)]

    for deg in range(1, degree + 1):
        for i in range(n_features):
            feat.append(X[:,i] ** deg)

    if degree >= 2:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                feat.append(X[:,i] * X[:, j])

    return np.vstack(feat).T

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        input_features = preprocess_input(request.form['ambient_temp'],
                                              request.form['module_temp'],
                                              request.form['irradiation']
                                              )
        
        if isinstance(model_data, dict) and model_data.get('type') == 'ridge':
            mean_f = model_data['mean']
            std_f = model_data['std']
            w = model_data['weights']
            deg = model_data['degree']
    
            X_std = (input_features - mean_f)/std_f
    
            X_poly = polynomial_feature_inf(X_std, deg)
    
            prediction = X_poly @ w
            prediction = prediction[0]
    
        elif model_data is not None:
            prediction = model_data.predict(input_features)[0]
    
        else:
            raise ValueError("Model failed to load on server startup.")
    
    
        final_prediction = max(0.0, round(float(prediction), 2))
    
        return render_template('results.html', prediction = final_prediction)

    except Exception as e:
        return render_template('results.html', prediction = f'Error: {str(e)}')


@app.route('/dashboard')
def dashboard():
    META_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'Results',
        'model_meta.json'
    )

    with open(META_PATH, 'r') as f:
        metrics = json.load(f)

    return render_template('dashboard.html', metrics=metrics)


if __name__ == '__main__':
    app.run(debug = True)


        





    


    







