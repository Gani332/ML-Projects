import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
#import swifter
#from langdetect import detect
#import seaborn as sns
#import spacy
#import en_core_web_sm

f = "RealvsFake.csv"

data_set = pd.read_csv(f, encoding='utf-8')

'''print(data_set.head())
print(data_set.info())'''

X_train, X_test, y_train, y_test = train_test_split(data_set.text, data_set.State, test_size=0.2, random_state=40)
count_vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True)


X_train_counts = count_vectorizer.fit_transform(X_train.values.astype('U'))
X_test_counts = count_vectorizer.transform(X_test.values.astype('U'))

mnb = MultinomialNB()
mnb.fit(X_train_counts, y_train)
y_predicted_counts = mnb.predict(X_test_counts)


def get_metrics(y_test, y_predicted):
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1


accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))
