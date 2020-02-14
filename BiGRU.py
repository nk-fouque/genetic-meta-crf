#!/usr/bin/python3
import sys
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

args = sys.argv
dict_words = dict()
dict_words['pad']=0
dict_words['oov']=1

dict_labels = dict()
dict_labels['O']=0

longest_line = 0

with open(args[1], "r") as f:
	corpus = []
	prediction = []
	corpus_current = []
	prediction_current = []
	for line in f:
		t = line.split()
		if len(t) == 0:
			longest_line=max(len(corpus_current),longest_line)
			corpus.append(corpus_current)
			corpus_current = []
			prediction.append(prediction_current)
			prediction_current = []
			continue
		if(t[0] not in dict_words):
			dict_words[t[0]]=len(dict_words)+1
		if(t[1] not in dict_labels):
			dict_labels[t[1]]=len(dict_labels)+1
		corpus_current.append(t[0])
		prediction_current.append(t[1])


'''
for s,l in zip(corpus, prediction):
	print(s)
	print(l)
'''

print(dict_words)
print(dict_labels)
print(longest_line)


#TODO construire les matrices

#TODO faire un réseau de neurones izy lol
