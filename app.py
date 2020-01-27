import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load, dump
import pandas as pd

app = Flask(__name__)

gaussian_model = load(r'D:\Heart Attack Detection\Models\gaussian_model.pkl')
knn_model = load(r'D:\Heart Attack Detection\Models\knn_model.pkl')
logreg_model = load(r'D:\Heart Attack Detection\Models\logreg_model.pkl')
perceptron_model = load(r'D:\Heart Attack Detection\Models\perceptron_model.pkl')
rf_model = load(r'D:\Heart Attack Detection\Models\rf_model.pkl')
sgd_model = load(r'D:\Heart Attack Detection\Models\sgd_model.pkl')


@app.route("/hello")
def hello():
    return "Hello"


@app.route("/")
def index():
    return "Index Page"


@app.route('/form')
def form():
    return render_template('form.html', my_style="/static/styles/form_style.css",
                           my_script="/static/scripts/form_validate.js")


@app.route("/gaussian_predict", methods=['POST'])
def gaussian_predict():
    data = request.form.to_dict()
    prediction = gaussian_model.predict(pd.DataFrame.from_dict([data]))
    print(prediction)
    output = prediction[0]
    if output == 0:
        return render_template('result.html', prediction="No")
    else:
        return render_template('result.html', prediction="Yes")


@app.route("/knn_predict", methods=['POST'])
def knn_predict():
    data = request.form.to_dict()
    prediction = gaussian_model.predict(pd.DataFrame.from_dict([data]))
    print(prediction)
    output = prediction[0]
    if output == 0:
        return render_template('result.html', prediction="No")
    else:
        return render_template('result.html', prediction="Yes")


if __name__ == "__main__":
    app.run(debug=True)
