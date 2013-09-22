import pandas as pd
import numpy as np
import nltk


def F1(pred_tags, true_tags):
    assert(isinstance(pred_tags,set))
    assert(isinstance(true_tags,set))
    
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

train = pd.read_csv('Train_10000.csv')

train['Tags'] = train['Tags'].apply(nltk.word_tokenize).apply(set)

print score([set(['javascript']) for i in range(len(train.Tags))], train.Tags)
print score([set(['javascript','c++','python','php','java']) for i in range(len(train.Tags))], train.Tags)

