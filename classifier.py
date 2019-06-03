#!/bin/python
import tarfile

from nltk import word_tokenize
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


class Classifier(object):
    """
    Outputs predictions for sentiment analysis
    """

    def __init__(self, tarfname='data/sentiment.tar.gz'): #modify
        # sentiment = self.read_files(tarfname)
        sentiment = self.read_files_input(tarfname)
        #full_data = sentiment.train_data + sentiment.dev_data ##output does not have dev data
        full_data = sentiment.train_data
        y_train = sentiment.trainy
        y_dev = sentiment.devy
        y_full = np.concatenate((y_train, y_dev))
        self.vect, self.clf = self.supervised_classifier(full_data, y_full)

    def predict_sentiment(self, review_data=""):
        X_test = self.vect.transform([review_data])
        y_pred = self.clf.predict_proba(X_test)[:,1]
        return y_pred

    def supervised_classifier(self, train_data, y_train):
        vect = CountVectorizer(ngram_range=(1,2), tokenizer=word_tokenize, max_df=1.0, min_df=3)
        X_train = vect.fit_transform(train_data)
        clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000)
        clf.fit(X_train, y_train)
        return vect, clf

    def read_files_input(self, filename, verbose=False):
        '''
        Input: A training test file
        '''


        class Data: pass
        sentiment = Data()
        tweet_train_label = []
        tweet_train_data = []
        tweet_dev_label = []
        tweet_dev_data = []


        with open(filename, encoding='utf8', errors='ignore') as f:
            next(f)
            for line in f:
                values = line.split()
                index, label, tweet = values[0], values[1], ' '.join(values[2:])
                tweet_train_label.append(label)
                tweet_train_data.append(tweet)

        sentiment.train_labels = tweet_train_label
        sentiment.train_data = tweet_train_data
        sentiment.dev_labels = tweet_dev_label
        sentiment.dev_data = tweet_dev_data

        sentiment.le = preprocessing.LabelEncoder()
        sentiment.le.fit(sentiment.train_labels)
        sentiment.target_labels = sentiment.le.classes_
        sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
        sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
        return sentiment

    def read_files(self, tarfname, verbose=False):
        """Read the training and development data from the sentiment tar file.
        The returned object contains various fields that store sentiment data, such as:

        train_data,dev_data: array of documents (array of words)
        train_labels,dev_labels: the true string label for each document (same length as data)

        The data is also preprocessed for use with scikit-learn, as:

        le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
        target_labels: List of labels (same order as used in le)
        trainy,devy: array of int labels, one for each document
        """
        def read_tsv(tar, fname):
            member = tar.getmember(fname)
            if verbose:
                print(member.name)
            tf = tar.extractfile(member)
            data = []
            labels = []
            for line in tf:
                line = line.decode("utf-8")
                (label,text) = line.strip().split("\t")
                labels.append(label)
                data.append(text)
            return data, labels

        tar = tarfile.open(tarfname, "r:gz")
        trainname = "train.tsv"
        devname = "dev.tsv"
        for member in tar.getmembers():
            if 'train.tsv' in member.name:
                trainname = member.name
            elif 'dev.tsv' in member.name:
                devname = member.name

        class Data: pass
        sentiment = Data()

        sentiment.train_data, sentiment.train_labels = read_tsv(tar, trainname)
        if verbose:
            print("-- train data")
            print(len(sentiment.train_data))

        sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
        if verbose:
            print("-- dev data")
            print(len(sentiment.dev_data))

        sentiment.le = preprocessing.LabelEncoder()
        sentiment.le.fit(sentiment.train_labels)
        sentiment.target_labels = sentiment.le.classes_
        sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
        sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
        tar.close()
        return sentiment

def usage():
    sys.stderr.write("""
    Usage: python classifier.py [review_text]
        Predict sentiment.\n""")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    review_data = " ".join(sys.argv[1:])
    tarfname = "data/SemEval2018-T3-train-taskA.txt"
    clf = Classifier(tarfname)
    y_pred = clf.predict_sentiment(review_data)
    print(y_pred)
