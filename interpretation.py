import itertools
from collections import defaultdict

from nltk import word_tokenize, TweetTokenizer

class Interpretation(object):
    """
    Interpretation of text. Handles visualizations.
    """
    def __init__(self, clf, model, text=""):
        clf_feature_names = clf.vect.get_feature_names()
        clf_coefs = clf.clf.coef_[0]
        self.vocab_dict = defaultdict(float, zip(clf_feature_names, clf_coefs))
        # self.pred = clf.predict_sentiment(text)
        # self.text = text
        if model == "sentiment":
            self.tokenized_text = word_tokenize(text)
        elif model == "sarcasm":
            tknz = TweetTokenizer()
            self.tokenized_text = tknz.tokenize(text)
        else:
            print("ERROR: {} model not recognized".format(model))
        self.compact_interpret = self.interpret_compact(self.tokenized_text)
        self.expanded_interpret = self.interpret_expanded(self.tokenized_text)

    def classify_token(self, token):
        # if token not in self.vocab_dict:
        if self.vocab_dict[token.lower()] == 0:
            return 'neutral'
        elif self.vocab_dict[token.lower()] > 0:
            return 'positive'
        else:
            return 'negative'

    def score(self, token):
        return self.vocab_dict[token]

    def decorate_text(text):
        return [(t, self.score(t), self.classify_token(t)) for t in text]

    def interpret_compact(self, text):
        def adjusted_score(left_token, token, right_token):
            return self.score(left_token)/2 + self.score(token) + self.score(right_token)/2
        lower_tokenized_text = [x.lower() for x in self.tokenized_text]
        lower_valid_text = list(valid_ngrams(lower_tokenized_text))
        valid_text = list(valid_ngrams(self.tokenized_text))
        v_text = [" "] + lower_valid_text
        compact_tokens = [t for i, t in enumerate(valid_text) if i % 2 == 0]
        compact_scores = [adjusted_score(v_text[i-1], v_text[i], v_text[i+1])
                          for i in range(len(v_text))
                          if i % 2 != 0]
        compact_sentiment = [classify_score(s) for s in compact_scores]
        compact_str_scores = [format_score(s) for s in compact_scores]
        return zip(compact_tokens, compact_str_scores, compact_sentiment)

    def interpret_expanded(self, text):
        def decorate_token(token):
            str_score = format_score(self.score(token))
            sentiment = self.classify_token(token)
            if " " in token:  # token is a bigram
                decorated_token = ("/{}/".format(token.replace(" ","_")), str_score, sentiment)
            else:
                decorated_token = (token, str_score, sentiment)
            return decorated_token
        valid_text = valid_ngrams(self.tokenized_text)
        return [decorate_token(t) for t in valid_text if self.score(t) != 0 or " " not in t]

def classify_score(score):
    # if token not in self.vocab_dict:
    if score == 0:
        return 'neutral'
    elif score > 0:
        return 'positive'
    else:
        return 'negative'

def format_score(score):
    return "{:.4f}".format(score) if score != 0 else "0"

def valid_ngrams(text):
    def flatten(iterables):
        return (elem for iterable in iterables for elem in iterable)
    bigrams = [" ".join((a, b)) for a, b in zip(text[:-1], text[1:])]
    bigrams.append("")
    valid = flatten(zip(text, bigrams))
    return valid
