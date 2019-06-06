# CSE 256 SP 19: Final Project: Text Classification

Interpret text classification model

## Getting Started

Run Flask app

		python app.py

## Files

* __app.py__: Flask app
* __classifier.py__: Sentiment Analysis and Sarcasm Detection models
* __interpretation.py__: logic for displaying interpretation in templates/interpretation.html
* __prediction.py__: logic for displaying prediction in templates/prediction.html
* __toptokens.py__: logic for displaying top 10 tokens in templates/topk.html
* __data\SemEval2018-T3_input_test_taskA.txt__: test data for sarcasm detection
* __data\SemEval2018-T3-train-taskA.txt__: train data for sarcasm detection
* __data\sentiment.tar.gz__: test data and train data for food reviews sentiment analysis
* __static/styles.css__: Formats webpage
* __static/Chart.min.js__: JavaScript visualization framework used for "Top 10 Tokens" graph
* __static/jquery.min.js__: Makes JavaScript easier
* __templates/index.html__: Main webpage
* __templates/interpret.html__: Dynamically visualize interpretation (via HTML embedded in index.html)
* __templates/predict.html__: Dynamically visualize prediction (via HTML embedded in index.html)
* __templates/topk.html__: Dynamically visualize "Top 10 Tokens" graph (via HTML embedded in index.html)
