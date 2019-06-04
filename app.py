import time

from flask import Flask, render_template, url_for, request
from interpretation import Interpretation
from classifier import SentimentClassifier, SarcasmClassifier


app = Flask(__name__)
sentiment_clf = SentimentClassifier()
sarcasm_clf = SarcasmClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interpret')
def interpret():
    text = request.args.get('jsdata')
    if True:
        interpretation = Interpretation(sentiment_clf, text)
    else:
        interpretation = Interpretation(sarcasm_clf, text)
    return render_template('interpret.html', interpretation=interpretation)

if __name__ == '__main__':
    app.run(debug=True)
