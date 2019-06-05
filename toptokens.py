import numpy as np


class TopTokens(object):
    """
    Outputs prediction and text.
    """
    def __init__(self, clf, model="", text=""):
        clf_pred = clf.predict_sentiment(text)
        neg_labels, neg_data = top_k(clf, text, 10, 'Negative')
        self.neg_labels = neg_labels #.append("")
        self.neg_data = neg_data #.append(0)
        self.neg_backgroundColor = ["#8b0000" for i in range(len(self.neg_data))]
        pos_labels, pos_data = top_k(clf, text, 10, 'Positive')
        self.pos_labels = pos_labels #.append("")
        self.pos_data = pos_data #.append(0)
        self.pos_backgroundColor = ["#006400" for i in range(len(self.pos_data))]

def top_k(clf, review=None, k=10, type='Top', verbose=False):
    """Return top-k predictive features"""
    clf_feature_names = clf.vect.get_feature_names()
    clf_coefs = clf.clf.coef_[0]

    if review is not None:
        X_review = clf.vect.transform([review])
        coef_indices = [idx for idx, x in enumerate(X_review.toarray()[0]) if x != 0]
        clf_coefs = clf_coefs[coef_indices]
        clf_feature_names = [clf_feature_names[i] for i in coef_indices]

    if type == 'Top':
        top_k = np.argsort(np.abs(clf_coefs))[-k:][::-1]
    elif type == 'Neutral':
        top_k = np.argsort(np.abs(clf_coefs))[:k]
    elif type == 'Positive':
        top_k = np.argsort(clf_coefs)[-k:][::-1]
    elif type == 'Negative':
        top_k = np.argsort(clf_coefs)[:k]
    else:
        print("Error: ")
        return

    top_k_coefs = np.array([clf_coefs[i] for i in top_k])
    top_k_words = np.array([clf_feature_names[i] for i in top_k])

    if type == 'Positive':
        top_k_words = top_k_words[top_k_coefs > 0]
        top_k_coefs = top_k_coefs[top_k_coefs > 0]
    elif type == 'Negative':
        top_k_words = top_k_words[top_k_coefs < 0]
        top_k_coefs = top_k_coefs[top_k_coefs < 0]

    if verbose:
        max_token_len = max([len(x) for x in clf_feature_names])
        print('-'*50)
        print('{} k={}'.format(type, k))
        print('-'*50)
        for i in top_k:
            print('{:{width}}  [{:.10f}]'.format(clf_feature_names[i], clf_coefs[i], width=max_token_len))
    return list(top_k_words), list(top_k_coefs)
