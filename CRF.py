#!/usr/bin/python3
import sys
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

args = sys.argv
with open(args[1], "r") as f:
	corpus = []
	current = []
	for line in f:
		t = line.split()
		if len(t) == 0:
			corpus.append(current)
			current = []
			continue
		current.append(line.split())

#print(corpus)

def word2features(sent, i):
	currentword = sent[i][0]

	features = {
		"word.lower()":currentword.lower(),#minus
		"word[−3:]": currentword[-3:],#suffix
		"word.isdigit()": currentword.isdigit(),
		"previousword": sent[i-1][0],
		"currentword": currentword
		}
	return features

def word2labels(sent, i):
	currentlabel = sent[i][1]
	print(([currentlabel]*5))
	return ([currentlabel]*5)


datas = corpus

X = [word2features(s, i) for s in datas for i in range(len(s))]#compute features
y = [word2labels(s, i) for s in datas for i in range(len(s))]#get target labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(len(X_train[0]))
print(len(y_train[0]))
crf = sklearn_crfsuite.CRF(algorithm="l2sgd",max_iterations=100)
crf.fit(X_train, y_train)
y_predict = crf.predict(X_test)
print(y_predict)
