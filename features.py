import csv
import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from treebank import TreebankWordTokenizer
from nltk import WhitespaceTokenizer
from nltk.stem import PorterStemmer
from joblib import Parallel, delayed
import scipy.sparse

def F1(pred_tags, true_tags):
    assert(isinstance(pred_tags,set))
    assert(isinstance(true_tags,set))

    if len(pred_tags) == 0 or len(true_tags) == 0:
        return 0
    
    intersect = pred_tags.intersection(true_tags)
    tp = float(len(intersect))
    fp = float(len(pred_tags - intersect))
    fn = float(len(true_tags - intersect))
    
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    if (p + r) > 0:
        return 2*p*r/(p + r)
    else:
        return 0.

def score(prediction, labels):
    
    f1_scores = []
    for pred, lab in zip(prediction, labels):
        f1_scores.append(F1(pred,lab))
    
    return np.mean(f1_scores)

def make_submission(title_vectorizer, classifiers, threshold, filename='submission.csv'):
    print "loading data..."
    test = pd.read_csv('Test.csv', usecols=['Id', 'Title', 'Tags'], nrows=100000)
    print "vectorizing titles..."
    X_test = title_vectorizer.transform(test['Title'])
    
    print "predicting tag probabilities..."
    Y_test = []
    for itag in range(ntags):
        Y_test.append(scipy.sparse.csr_matrix(classifiers[itag].predict_proba(X_test)[:,1] > threshold))
    Y_test = scipy.sparse.vstack(Y_test).T.tocsr()

    print "choosing output tags..."
    tags = []
    for isample in range(Y_test.shape[0]):
        if isample%10000 == 0:
            print isample
        tags.append(set(tag_vectorizer.inverse_transform(Y_test[isample])[0]))
    
    with open(filename, 'w') as fd:
        writer = csv.writer(fd, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['Id','Tags'])
        for id, tag_set in zip(test.Id, tags):
            writer.writerow([id, ' '.join(tag_set)])


print "loading data..."
data = pd.read_csv('Train.csv', usecols=['Title', 'Tags'], nrows=10000)
#data.Body = data.Body.apply(nltk.clean_html)

ntrain = 0.7*data.Title.count()
train, validate = data[:ntrain], data[ntrain:]

stemmer = PorterStemmer()

def tokenizer(s):
    words = nltk.word_tokenize(s)
    return [stemmer.stem_word(w) for w in words]

print "vectorizing titles..."
title_vectorizer = CountVectorizer(#stop_words=set(nltk.corpus.stopwords.words()), 
                                   stop_words='english',
                                   #tokenizer=nltk.word_tokenize, 
                                   #tokenizer=TreebankWordTokenizer().tokenize,
                                   tokenizer=tokenizer, 
                                   strip_accents='unicode',
                                   max_features=30000)
X_train = title_vectorizer.fit_transform(train['Title'])

print "number of title features:", X_train.shape[1]

print "vectorizing tags..."
tag_vectorizer = CountVectorizer(stop_words='english', 
                                 tokenizer=WhitespaceTokenizer().tokenize,
                                 binary=True,
                                 max_features=5000)
Y_train = tag_vectorizer.fit_transform(train['Tags'])
ntags = Y_train.shape[1]

print "number of tags:", ntags

def train_classifier(Y):
    classifier = LogisticRegression()
    classifier.fit(X_train,Y)
    return classifier

print "training classifiers..."
#classifiers = []
classifiers = Parallel(n_jobs=4, verbose=40)(delayed(train_classifier)(np.ravel(Y_train[:,itag].todense())) for itag in range(Y_train.shape[1]))

'''
for itag in range(Y_train.shape[1]):
    if itag%1000 == 0:
        print itag, Y_train.shape[1]
    classifiers.append(LogisticRegression())
    classifiers[-1].fit(X_train,np.ravel(Y_train[:,itag].todense()))
'''

X_validate = title_vectorizer.transform(validate['Title'])

threshold = 0.1

print "predicting tag probabilities..."
Y_hat = []
for itag in range(ntags):
    Y_hat.append(scipy.sparse.csr_matrix(classifiers[itag].predict_proba(X_validate)[:,1] > threshold))
Y_hat = scipy.sparse.vstack(Y_hat).T.tocsr()

print "choosing output tags..."
tags = []
for isample in range(Y_hat.shape[0]):
    tags.append(set(tag_vectorizer.inverse_transform(Y_hat[isample])[0]))

print "scoring..."
print score(tags, validate.Tags.apply(WhitespaceTokenizer().tokenize).apply(set))

make_submission(title_vectorizer=title_vectorizer, classifiers=classifiers, threshold=threshold)


