class Prediction(object):
    """
    Outputs prediction and text.
    """
    def __init__(self, clf, model="", text=""):
        clf_pred = float(clf.predict_sentiment(text))
        self.neg_pred = format_prediction(1 - clf_pred)
        self.pos_pred = format_prediction(clf_pred)
        if model == "sentiment":
            if clf_pred < 0.5:
                self.classification = "Negative"
                self.sentiment = "red"
            else:
                self.classification = "Positive"
                self.sentiment = "green"
        elif model == "sarcasm":
            if clf_pred < 0.5:
                self.classification = "Not Sarcastic"
                self.sentiment = "red"
            else:
                self.classification = "Sarcastic"
                self.sentiment = "green"
        self.text = text

def format_prediction(pred):
    return "{:.1f}%".format(100*float(pred))
