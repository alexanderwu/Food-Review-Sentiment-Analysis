import json

from flask import Flask, render_template, request
from prediction import Prediction
from interpretation import Interpretation
from toptokens import TopTokens
from classifier import SentimentClassifier, SarcasmClassifier


app = Flask(__name__)
sentiment_clf = SentimentClassifier()
sarcasm_clf = SarcasmClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    model = request.args.get('model')
    text = request.args.get('text')
    if model == "sentiment":
        prediction = Prediction(sentiment_clf, model, text)
    elif model == "sarcasm":
        prediction = Prediction(sarcasm_clf, model, text)
    else:
        print("ERROR: {} model not recognized".format(model))
    return render_template('prediction.html', prediction=prediction)


@app.route('/interpret')
def interpret():
    model = request.args.get('model')
    text = request.args.get('text')
    if model == "sentiment":
        interpretation = Interpretation(sentiment_clf, model, text)
    elif model == "sarcasm":
        interpretation = Interpretation(sarcasm_clf, model, text)
    else:
        print("ERROR: {} model not recognized".format(model))
    return render_template('interpret.html', interpretation=interpretation)

@app.route('/topk')
def topk():
    model = request.args.get('model')
    text = request.args.get('text')
    if model == "sentiment":
        toptokens = TopTokens(sentiment_clf, model, text)
    elif model == "sarcasm":
        toptokens = TopTokens(sarcasm_clf, model, text)
    else:
        print("ERROR: {} model not recognized".format(model))

    return render_template('topk.html', tt=toptokens)

if __name__ == '__main__':
    app.run(debug=True)
