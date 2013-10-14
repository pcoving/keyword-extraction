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

import multiprocessing
import ctypes


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
    test = pd.read_csv('Test.csv', usecols=['Id', 'Title', 'Tags'])
    print "vectorizing titles..."
    X_test = title_vectorizer.transform(test['Title'])
    
    print "predicting tag probabilities..."
    Y_test = []
    for itag in range(ntags):
        prob = classifiers[itag].predict_proba(X_test)[:,1]
        prob[prob < threshold] = 0.
        Y_test.append(scipy.sparse.csr_matrix(prob))

    Y_test = scipy.sparse.vstack(Y_test).T.tocsr()

    print "choosing output tags..."
    tags = []
    for isample in range(Y_test.shape[0]):
        tmp = np.ravel(Y_test[isample].todense())
        tmp[tmp.argsort()[::-1][2:]] = 0.
        tags.append(set(tag_vectorizer.inverse_transform(tmp)[0]))
    
    '''
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
    '''

    with open(filename, 'w') as fd:
        writer = csv.writer(fd, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['Id','Tags'])
        for id, tag_set in zip(test.Id, tags):
            writer.writerow([id, ' '.join(tag_set)])

print "loading data..."
nrows = 100000
data = pd.read_csv('Train.csv', usecols=['Title', 'Tags'], nrows=nrows)
#data.Body = data.Body.apply(nltk.clean_html)

#data.Title += ' ' + data.Body

print 'number of rows:', nrows

n_jobs = 4
verbose = 3

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
                                   binary=True,
                                   max_features=30000)
#title_vectorizer = HashingVectorizer(stop_words='english',
#                                     tokenizer=tokenizer, 
#                                     strip_accents='unicode')
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
    classifier = LogisticRegression(C=1.0)
    classifier.fit(X_train,Y)
    return classifier

print "training classifiers..."
pool = multiprocessing.Pool(processes=4)
classifiers = pool.map(train_classifier, (np.ravel(Y_train[:,itag].todense()) for itag in range(ntags)))

#classifiers = []
#classifiers = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(train_classifier)(np.ravel(Y_train[:,itag].todense())) for itag in range(ntags))



'''
for itag in range(Y_train.shape[1]):
    if itag%1000 == 0:
        print itag, Y_train.shape[1]
    classifiers.append(LogisticRegression())
    classifiers[-1].fit(X_train,np.ravel(Y_train[:,itag].todense()))
'''

X_validate = title_vectorizer.transform(validate['Title'])

threshold = 0.0001

def predict_proba(classifier):
    prob = classifier.predict_proba(X_validate)[:,1]
    prob[prob < threshold] = 0.
    return scipy.sparse.csr_matrix(prob)

print "predicting tag probabilities..."
Y_hat = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(predict_proba)(classifiers[itag]) for itag in range(ntags))

'''
Y_hat = []
for itag in range(ntags):
    prob = classifiers[itag].predict_proba(X_validate)[:,1]
    prob[prob < threshold] = 0.
    Y_hat.append(scipy.sparse.csr_matrix(prob))
'''

Y_hat = scipy.sparse.vstack(Y_hat).T.tocsr()

# train and predict number of tags...
z = np.asarray(np.sum(Y_train.todense(),axis=1).ravel()).ravel()

clf = LogisticRegression(C=0.02)
clf.fit(X_train, z)
zhat = clf.predict(X_validate)

def choose_tags(y, idx):
    tmp = np.ravel(y.todense())
    #tmp[tmp.argsort()[::-1][2:]] = 0.
    tmp[tmp.argsort()[::-1][zhat[idx]:]] = 0.
    return set(tag_vectorizer.inverse_transform(tmp)[0])
    #return set(tag_vectorizer.inverse_transform(np.ravel(y.todense()) > threshold)[0])

print "choosing output tags..."
tags = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(choose_tags)(Y_hat[isample], isample) for isample in range(Y_hat.shape[0]))

'''
tags = []
for isample in range(Y_hat.shape[0]):
    tmp = np.ravel(Y_hat[isample].todense())
    tmp[tmp.argsort()[::-1][2:]] = 0.
    tags.append(set(tag_vectorizer.inverse_transform(tmp)[0]))
'''

print "scoring..."
print score(tags, validate.Tags.apply(WhitespaceTokenizer().tokenize).apply(set))


pool.close()

#make_submission(title_vectorizer=title_vectorizer, classifiers=classifiers, threshold=threshold)

'''
tmp = classifiers[itag].predict_proba(X_validate)[:,1]
idx = tmp.argsort()[::-1]
tmp[:] = 0.
tmp[idx[:3]] = 1
Y_hat.append(scipy.sparse.csr_matrix(tmp))
'''
