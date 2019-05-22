import time

from flask import Flask, render_template, url_for, request
from interpretation import Interpretation


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interpret')
def interpret():
    text = request.args.get('jsdata')
    interpretation = Interpretation(text)
    return render_template('interpret.html', interpretation=interpretation)

if __name__ == '__main__':
    app.run(debug=True)
