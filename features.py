ey
import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from treebank import TreebankWordTokenizer
from nltk import WhitespaceTokenizer
from nltk.stem import PorterStemmer
from joblib import Parallel, delayed
import scipy.sparse

import multiprocessing
import ctypes

import cloud
import pickle
import math
from collections import defaultdict
import hashlib
import time
import re

import warnings
warnings.filterwarnings('ignore')

from random import random

# need to optimize BOTH wrt regularization and min_freq of tokens

# ratio exact tags: 0.5732979625368232

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

def choose_ntags(X_train, Y_train, X_test):    
    # train and predict number of tags...
    ntags_train = np.asarray(Y_train.sum(axis=1)).ravel()

    clf = LogisticRegression(C=0.02)
    clf.fit(X_train, ntags_train)
    return clf.predict(X_test)

'''
X_train = pickle.load(open('X_train.p'))
Y_train = pickle.load(open('Y_train.p'))
X_validate = pickle.load(open('X_validate.p'))
Y_validate = pickle.load(open('Y_validate.p'))
'''

def make_planes(hash_size, dim):
    planes = np.random.rand(hash_size, dim)
    planes[planes <= 0.5] = -1
    planes[planes > 0.5] = 1
    return planes

def lsh(vectors, planes):
    result = vectors.dot(planes.T)
    mask = result <= 0
    result = np.power(2, np.indices(result.shape)[1])
    result[mask] = 0
    return np.sum(result, axis=1)

'''
planes = make_planes(22, X_train.shape[1])
train_hash = lsh(X_train, planes)

hash_list = defaultdict(list)
for idx, val in enumerate(train_hash):
    hash_list[val].append(idx)

test_hash = lsh(X_test, planes)

''' 
''''
    if (X_test[idx, :].sum() == 0 or len(hash_list[test_hash[idx]]) == 0):
        Y_hat.append(['javascript', 'c#', 'python', 'php', 'java'])
    else:
        tmp = X_train[hash_list[test_hash[idx]], :].dot(X_test[idx, :].T)
        if (idx%10000 == 0):
            print idx, tmp.max()/float(X_test[idx,:].sum())
        match = hash_list[test_hash[idx]][tmp.todense().argmax()]
        Y_hat.append(train['Tags'][match].split(' '))
    '''''
''''
with open('submission.csv', 'w') as fd:
    writer = csv.writer(fd, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['Id','Tags'])
    for id, tags in zip(test.Id, Y_hat):
        writer.writerow([id, ' '.join(tags)])
'''''
''''
X_train = title_vectorizer.transform(train['Title'])
X_test = title_vectorizer.transform(test['Title'])
Y_hat = []
for idx in range(X_test.shape[0]):
    tmp = X_train.dot(X_test[idx,:].T)
    print idx, tmp.max()
    match = tmp.todense().argmax()
    #print train['Title'][match], '\n', test['Title'][idx], '\n', train['Tags'][match]
    Y_hat.append(train['Tags'][match].split(' '))

def doit(idx):
    tmp = X_train.dot(X_test[idx,:].T)
    print idx, tmp.max()
    match = tmp.todense().argmax()
    #print train['Title'][match], '\n', test['Title'][idx], '\n', train['Tags'][match]
    return train['Tags'][match].split(' ')
'''

'''
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
'''

def split_data(array, nworkers=20):
    chunk_size = math.ceil(float(array.shape[0])/nworkers)
    splat = []
    for iworker in range(nworkers):
        splat.append(array[iworker*chunk_size:(iworker+1)*chunk_size])
    return splat


def train_on_cloud(nworkers=20):
    chunk_size = float(ntags)/nworkers
    
    def train_classifier(tag_lower, tag_upper, iworker):
        time.sleep(120*iworker)
        print "training classifier..."
        classifier = LogisticRegression(C=0.5, tol=0.1, penalty='l1')
        Y_train = pickle.load(open('/bucket/Y_train', 'rb'))
        X_train = pickle.load(open('/bucket/X_train_new', 'rb'))
        X_test = pickle.load(open('/bucket/X_test_new', 'rb'))
        
        prob_list = []
        for itag in range(tag_lower, tag_upper):
            print itag, chunk_size
            classifier.fit(X_train, np.ravel(Y_train[:,itag].todense()))
            prob = classifier.predict_proba(X_test)[:,1]
            prob[prob < 0.01] = 0.
            prob_list.append(scipy.sparse.csr_matrix(prob))
        Y_hat = scipy.sparse.vstack(prob_list).T.tocsr()
        pickle.dump(Y_hat, open('/bucket/' + 'Y_hat_' + str(iworker), 'wb'))
        #cloud.bucket.putf(pickle.dumps(Y_hat), 'Y_hat_' + str(itag))
    

    return cloud.map(train_classifier, [int(idx*chunk_size) for idx in range(nworkers)], [int((idx+1)*chunk_size) for idx in range(nworkers)], range(nworkers), _type='f2')

def retrieve_results(nworkers=20):
    Y_hat_list = []
    for iworker in range(nworkers):
        print iworker
        Y_hat_list.append(pickle.loads(cloud.bucket.getf('Y_hat_' + str(iworker)).read()))
    return scipy.sparse.hstack(Y_hat_list).tocsr()


'''
jobs = train_on_cloud()
Y_hat = scipy.sparse.hstack(cloud.result(jobs)).tocsr()

vectorizing titles (fit)...
vectorizing titles (transform)...
22920797
0.0005 l1 0.821939096905
0.001 l1 0.822088581785
0.005 l1 0.820336672078
0.01 l1 0.8208794384
0.05 l1 0.821900085126
0.1 l1 0.82221343457
0.0005 l2 0.822650440352
0.001 l2 0.822602804443
0.005 l2 0.822859736019
0.01 l2 0.822916208251
0.05 l2 0.822722824101
0.1 l2 0.822416010599
0.822916208251

199300263
0.001 l1 0.823493078261
0.005 l1 0.825171929494
0.01 l1 0.825690939679
0.05 l1 0.826000604655
0.1 l1 0.825429356484
0.5 l1 0.822919552945
0.001 l2 0.826609768123
0.005 l2 0.826296039447
0.01 l2 0.825901093331
0.05 l2 0.824299777262

'''

def predict_ntags(nrows, which):
    train = pd.read_csv('Train.csv', usecols=['Title', 'Tags', 'Body'], nrows=nrows)
    train.Body = train.Body.apply(nltk.clean_html)
    train['TitleBody'] = train.Title + ' ' + train.Body
    train['Count'] = train.Tags.apply(lambda s: len(s.split(' ')))
    
    ntrain = 0.7*train.Title.count()
    train, validate = train[:ntrain], train[ntrain:]
    
    stemmer = PorterStemmer()
    
    def tokenizer(s):
        s = re.sub("[\&)(',\?\!]", "", s.lower())
        s = re.sub("[\/,-]", " ", s)
        s = re.sub("[\.;:]\s", " ", s)
        words = nltk.word_tokenize(s)
        try:
            words = [stemmer.stem_word(w) for w in words]
        except:
            print 'exception'
            pass
        return words
    
    print "vectorizing titles (fit)..."
    title_vectorizer = CountVectorizer(stop_words='english',
                                       tokenizer=tokenizer, 
                                       strip_accents='unicode',
                                       binary=True,
                                       min_df=5)
    title_vectorizer.fit(train['Title'])
    print "vectorizing titles (transform)..."
    if which == 'Title':
        X_train = title_vectorizer.transform(train['Title'])
        X_validate = title_vectorizer.transform(validate['Title'])
    else:
        X_train = title_vectorizer.transform(train['TitleBody'])
        X_validate = title_vectorizer.transform(validate['TitleBody'])
    
    print X_train.nnz
    best_tags = []
    best_score = 0
    for penalty in ['l1', 'l2']:
        for C in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            clf = LogisticRegression(C=C, penalty=penalty)
            clf.fit(X_train, train['Count'])
            Y_hat = clf.predict(X_validate)
            pred_tags = [set(range(val)) for val in Y_hat]
            #pred_tags = [set([val2 + 10*int(round(random())) for val2 in range(val)]) for val in Y_hat]
            tags = [set(range(val)) for val in validate['Count']]
            my_score = score(pred_tags, tags)
            print C, penalty, my_score
            if my_score > best_score:
                best_score = my_score
                best_tags = Y_hat
    print best_score
    return best_tags


def load_and_vectorize(nrows=None, path='Train.csv', n_title_min=5, n_tag_min=5):
    print "loading data..."
    data = pd.read_csv('Train.csv', usecols=['Title', 'Tags', 'Body'], nrows=nrows)
    data.Body = data.Body.apply(nltk.clean_html)
    data['TitleBody'] = data.Title + ' ' + data.Body
    
    print 'number of rows:', data.shape[0]
    
    ntrain = 0.7*data.Title.count()
    train, validate = data[:ntrain], data[ntrain:]
    
    stemmer = PorterStemmer()
    
    def tokenizer(s):
        s = re.sub("[\&)(',\?\!]", "", s.lower())
        s = re.sub("[\/,-]", " ", s)
        s = re.sub("[\.;:]\s", " ", s)
        words = nltk.word_tokenize(s)
        return [stemmer.stem_word(w) for w in words]
            
    print "vectorizing titles (fit)..."
    title_vectorizer = CountVectorizer(#stop_words=set(nltk.corpus.stopwords.words()), 
        stop_words='english',
        #tokenizer=nltk.word_tokenize, 
        #tokenizer=TreebankWordTokenizer().tokenize,
        tokenizer=tokenizer, 
        strip_accents='unicode',
        binary=True,
        min_df=n_title_min)
        #max_features=30000)
    #vectorizer = HashingVectorizer(stop_words='english',
    #                               tokenizer=tokenizer, 
    #                               strip_accents='unicode',
    #                               binary=True)
    title_vectorizer.fit(train['Title'])
    print "vectorizing titles (transform)..."
    #X_train = title_vectorizer.transform(train['TitleBody'])
    X_train = title_vectorizer.transform(train['Title'])
    #pickle.dump(title_vectorizer, open('title_vectorizer.p', 'wb'))
    #title_vectorizer = pickle.load(open('title_vectorizer.p', 'rb'))
    #pickle.dump(tag_vectorizer, open('tag_vectorizer.p', 'wb'))
    #tag_vectorizer = pickle.load(open('tag_vectorizer.p', 'rb'))

    print "number of title features:", X_train.shape[1]

    print "vectorizing tags..."
    #tokenizer = WhitespaceTokenizer().tokenize
    def tokenizer(s):
        return WhitespaceTokenizer().tokenize(s)
    
    tag_vectorizer = CountVectorizer(tokenizer=tokenizer,
                                     binary=True,
                                     min_df=n_tag_min)
                                     #max_features=5000)
    Y_train = tag_vectorizer.fit_transform(train['Tags'])
    
    X_validate = title_vectorizer.transform(validate['Title'])
    #X_validate = title_vectorizer.transform(validate['TitleBody'])
    

    Y_validate = validate.Tags.apply(WhitespaceTokenizer().tokenize).apply(set)

    X_train, Y_train, X_validate = X_train.tocsr(), Y_train.tocsr(), X_validate.tocsr()
    
    return X_train, Y_train, X_validate, Y_validate, title_vectorizer, tag_vectorizer

def make_submission(tags, pred_tags, filename='submission.csv'):
    print 'making submission...'
    with open(filename, 'w') as fd:
        writer = csv.writer(fd, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(['Id','Tags'])
        
        for id, actual, pred in zip(range(6034196,8047533), tags, pred_tags):
            if (len(actual) > 0):
                writer.writerow([id, ' '.join(actual)])
            else:
                writer.writerow([id, ' '.join(pred)])

''''
X_train, Y_train, X_validate, Y_validate, title_vectorizer, tag_vectorizer = load_and_vectorize(nrows=10000)
pickle.dump(X_train, open('X_train.p', 'w'))
pickle.dump(Y_train, open('Y_train.p', 'w'))
pickle.dump(X_validate, open('X_validate.p', 'w'))
pickle.dump(Y_validate, open('Y_validate.p', 'w'))

cloud.bucket.putf(pickle.dumps(X_train), 'X_train')
print 'done'
cloud.bucket.putf(pickle.dumps(Y_train), 'Y_train')
print 'done'
cloud.bucket.putf(pickle.dumps(X_test), 'X_test')
print 'done'
cloud.bucket.putf(pickle.dumps(X_validate), 'X_validate')
'''

'''
pickle.dump(title_vectorizer, open('title_vectorizer.p', 'wb'))
pickle.dump(tag_vectorizer, open('tag_vectorizer.p', 'wb'))

pickle.dump(X_train, open('X_train.p', 'w'))
pickle.dump(Y_train, open('Y_train.p', 'w'))
pickle.dump(X_test, open('X_test.p', 'w'))
'''

'''
train = pd.read_csv('Train.csv', usecols=['Title', 'Body', 'Tags'])
test = pd.read_csv('Test.csv', usecols=['Title', 'Body', 'Id'])
train_dict = {}
for idx, title in enumerate(train['Body']):
    if (idx%100000 == 0): print idx
    titlebody = train['Body'][idx] + ' ' + train['Title'][idx]
    train_dict[hashlib.md5(titlebody).hexdigest()] = idx

tags = []
for idx, title in enumerate(test['Body']):
    if (idx%100000 == 0): print idx
    titlebody = hashlib.md5(test['Body'][idx] + ' ' + test['Title'][idx]).hexdigest()
    if (train_dict.has_key(titlebody)):
        tags.append(train['Tags'][train_dict[titlebody]].split(' '))
    else:
        tags.append([])
'''

# vanilla titles vectorized: 0.377317460317
# without stemmer: 0.373179047619
# vanilla titles+body vectorized: 0.391158095238
# non-binary counts: 0.374696825397
# extra regex: 0.382206507936

n_jobs = 4
verbose = 1

'''
stemmer = PorterStemmer()
    
def tokenizer(s):
    words = nltk.word_tokenize(s)
    return [stemmer.stem_word(w) for w in words]

title_vectorizer = pickle.load(open('title_vectorizer.p', 'rb'))
tag_vectorizer = pickle.load(open('tag_vectorizer.p', 'rb'))

X_train = pickle.load(open('X_train.p', 'r'))
Y_train = pickle.load(open('Y_train.p', 'r'))
X_test = pickle.load(open('X_test.p', 'r'))

tags = pickle.load(open('tags.p', 'r'))
ntags_test = pickle.load(open('ntags_test.p', 'r'))


#ntrain = 100000
ntrain = X_train.shape[0]
ntitles = 60000
ntags = 20000
print "number of titles, tags:", ntitles, ntags

title_mask = np.asarray(X_train[:ntrain].sum(axis=0))[0].argsort()[::-1][:ntitles]
tag_mask = np.asarray(Y_train[:ntrain].sum(axis=0))[0].argsort()[::-1][:ntags]

X_train = X_train[:ntrain, title_mask]
Y_train = Y_train[:ntrain, tag_mask]

X_test = X_test[:, title_mask]
'''

X_train, Y_train, X_test, Y_test, title_vectorizer, tag_vectorizer = load_and_vectorize(nrows=100000, n_title_min=5, n_tag_min=5)
ntags_total = len(tag_vectorizer.vocabulary_)
ntags = Y_train.shape[1]

tol = 0.1
for n_title_min in [5]:
    for n_tag_min in [5]:
        for C in [1.0]:
            for penalty in ['l1']:
            #for tol in [1e-1]:
                def train_classifier(Y):
                    classifier = LogisticRegression(C=C, penalty=penalty, tol=tol)
                    #classifier = BernoulliNB()
                    #classifier = MultinomialNB()
                    classifier.fit(X_train,Y)
                    return classifier

                '''
                def train_classifier(itag):
                    classifier = LogisticRegression(C=1.0)
                    Y = pickle.load(open('/bucket/Y_train'))
                    X = pickle.load(open('/bucket/X_train'))
                    classifier.fit(X,np.ravel(Y[:,itag].todense()))
                    X = pickle.load(open('/bucket/X_validate'))
                    prob = classifier.predict_proba(X)[:,1]
                    prob[prob < 0.001] = 0.
                    return scipy.sparse.csr_matrix(prob)
                '''

                #print "training classifiers..."
                #pool = multiprocessing.Pool(processes=6)
                #classifiers = pool.map(train_classifier, (np.ravel(Y_train[:,itag].todense()) for itag in range(ntags)))
                #pool.close()

                #classifiers = []
                classifiers = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(train_classifier)(np.ravel(Y_train[:,itag].todense())) for itag in range(ntags))

                '''
                for itag in range(Y_train.shape[1]):
                    if itag%1000 == 0:
                        print itag, Y_train.shape[1]
                    classifiers.append(LogisticRegression())
                    classifiers[-1].fit(X_train,np.ravel(Y_train[:,itag].todense()))
                '''

                threshold = 0.01

                def predict_proba(classifier):
                    prob = classifier.predict_proba(X_test)[:,1]

                    prob[prob < threshold] = 0.
                    return scipy.sparse.csr_matrix(prob)

                print "predicting tag probabilities..."
                #pool = multiprocessing.Pool(processes=4)
                #Y_hat = pool.map(predict_proba, (classifiers[itag] for itag in range(ntags)))
                #pool.close()

                Y_hat = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(predict_proba)(classifiers[itag]) for itag in range(ntags))

                '''
                Y_hat = []
                for itag in range(ntags):
                    prob = classifiers[itag].predict_proba(X_validate)[:,1]
                    prob[prob < threshold] = 0.
                    Y_hat.append(scipy.sparse.csr_matrix(prob))
                '''

                Y_hat = scipy.sparse.vstack(Y_hat).T.tocsr()


                '''
                def choose_tags(y, idx):
                    if (idx%100000 == 0): print idx
                       if len(tags[idx]) == 0:
                        indicies = y.nonzero()[1][y.data.argsort()[::-1][:ntags_test[idx]]]
                        tmp = scipy.sparse.csr_matrix((ntags_total,1))
                        tmp[tag_mask[indicies]] = 1
                        return tmp

                ##      y = np.ravel(y.todense())
                        #tmp[tmp.argsort()[::-1][2:]] = 0.
                ##        tmp = np.zeros(ntags_total)
                ##        for i in y.argsort()[::-1][:ntags_test[idx]]:
                ##            tmp[tag_mask[i]] = 1
                ##        return set(tag_vectorizer.inverse_transform(tmp)[0])

                    else:
                        return scipy.sparse.csr_matrix((ntags_total,1))
                        #return set(tag_vectorizer.inverse_transform(np.ravel(y.todense()) > threshold)[0])
                '''

                print "choosing output tags..."
                #pool = multiprocessing.Pool(processes=4)
                #tags = pool.map(choose_tags, ((Y_hat[isample], isample) for isample in range(Y_hat.shape[0])))
                #pool.close()

                #pred_tags = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(choose_tags)(Y_hat[isample], isample) for isample in range(Y_hat.shape[0]))
                #pred_tags = [list(t) for t in tag_vectorizer.inverse_transform(scipy.sparse.hstack(pred_tags).T)]

                pred_tags = scipy.sparse.lil_matrix((Y_hat.shape[0], ntags_total))
                for idx in range(Y_hat.shape[0]):
                    if (idx%100000 == 0): 
                        print idx
                    #if len(tags[idx]) == 0:
                    if True:
                        #num_tags = ntags_test[idx]
                        num_tags = 2
                        indicies = Y_hat[idx].nonzero()[1][Y_hat[idx].data.argsort()[::-1][:num_tags]]
                        #pred_tags[idx,tag_mask[indicies]] = 1
                        pred_tags[idx,indicies] = 1
                        
                pred_tags = [set(t) for t in tag_vectorizer.inverse_transform(pred_tags.tocsr())]

                #make_submission(tags, pred_tags, filename='submission.csv')


                ''''
                tags = []
                for isample in range(Y_hat.shape[0]):
                    tmp = np.ravel(Y_hat[isample].todense())
                    tmp[tmp.argsort()[::-1][2:]] = 0.
                    tags.append(set(tag_vectorizer.inverse_transform(tmp)[0]))
                '''

                #print "scoring..."
                print '***', n_title_min, n_tag_min, C, penalty, tol, score(pred_tags, Y_test)


                #make_submission(title_vectorizer=title_vectorizer, classifiers=classifiers, threshold=threshold)


pred_tags = scipy.sparse.lil_matrix((Y_hat.shape[0], ntags_total))
for idx in range(Y_hat.shape[0]):
    if (idx%100000 == 0):
        print idx
        #if len(tags[idx]) == 0:
    if True:
        num_tags = ntags_test[idx]
        #num_tags = 2
        indicies = Y_hat[idx].nonzero()[1][Y_hat[idx].data.argsort()[::-1][:num_tags]]
        #pred_tags[idx,tag_mask[indicies]] = 1
        pred_tags[idx,indicies] = 1

pred_tags = [set(t) for t in tag_vectorizer.inverse_transform(pred_tags.tocsr())]

make_submission(tags, pred_tags, filename='submission.csv')

