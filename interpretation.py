from nltk import word_tokenize
from sklearn.externals import joblib

class Interpretation(object):
    """
    Interpretation of text
    """
    def __init__(self, text=""):
        self.text = text
        self.wt_text = word_tokenize(text)
