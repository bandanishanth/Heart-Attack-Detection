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
    return render_template('index.html', my_style="/static/styles/form_style.css")


@app.route("/knn_form")
def knn_form():
    return render_template('knn_form.html', my_style="/static/styles/form_style.css")


@app.route("/gaussian_form")
def gaussain_form():
    return render_template('gaussian_form.html', my_style="/static/styles/form_style.css")


@app.route("/logreg_form")
def logreg_form():
    return render_template('logreg_form.html', my_style="/static/styles/form_style.css")


@app.route("/rf_form")
def rf_form():
    return render_template('rf_form.html', my_style="/static/styles/form_style.css")


@app.route("/perceptron_form")
def perceptron_form():
    return render_template('perceptron_form.html', my_style="/static/styles/form_style.css")


@app.route("/sgd_form")
def sgd_form():
    return render_template('sgd_form.html', my_style="/static/styles/form_style.css")


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


@app.route("/logreg_predict", methods=['POST'])
def logreg_predict():
    data = request.form.to_dict()
    prediction = logreg_model.predict(pd.DataFrame.from_dict([data]))
    print(prediction)
    output = prediction[0]
    if output == 0:
        return render_template('result.html', prediction="No")
    else:
        return render_template('result.html', prediction="Yes")


@app.route("/perceptron_predict", methods=['POST'])
def perceptron_predict():
    data = request.form.to_dict()
    prediction = perceptron_model.predict(pd.DataFrame.from_dict([data]))
    print(prediction)
    output = prediction[0]
    if output == 0:
        return render_template('result.html', prediction="No")
    else:
        return render_template('result.html', prediction="Yes")


@app.route("/rf_predict", methods=['POST'])
def rf_predict():
    data = request.form.to_dict()
    prediction = rf_model.predict(pd.DataFrame.from_dict([data]))
    print(prediction)
    output = prediction[0]
    if output == 0:
        return render_template('result.html', prediction="No")
    else:
        return render_template('result.html', prediction="Yes")


@app.route("/sgd_predict", methods=['POST'])
def sgd_predict():
    data = request.form.to_dict()
    prediction = sgd_model.predict(pd.DataFrame.from_dict([data]))
    print(prediction)
    output = prediction[0]
    if output == 0:
        return render_template('result.html', prediction="No")
    else:
        return render_template('result.html', prediction="Yes")


if __name__ == "__main__":
    app.run(debug=True)
