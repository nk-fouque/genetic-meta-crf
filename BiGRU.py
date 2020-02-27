#!/usr/bin/python3
import sys

import numpy as np
from keras import Input, Model
from keras.layers import Bidirectional, Embedding, TimeDistributed, Dropout, CuDNNLSTM
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if __name__ == '__main__':
    args = sys.argv
    dict_words = dict()
    dict_words['pad'] = 0
    dict_words['oov'] = 1

    dict_labels = dict()
    dict_labels['O'] = 0

    longest_line = 0

    with open(args[1], "r") as f:
        corpus = []
        prediction = []
        corpus_current = []
        prediction_current = []
        for line in f:
            t = line.split()
            if len(t) == 0:
                longest_line = max(len(corpus_current), longest_line)
                corpus.append(corpus_current)
                corpus_current = []
                prediction.append(prediction_current)
                prediction_current = []
                continue
            if (t[0] not in dict_words):
                dict_words[t[0]] = len(dict_words) + 1
            if (t[1] not in dict_labels):
                dict_labels[t[1]] = len(dict_labels) + 1
            corpus_current.append(t[0])
            prediction_current.append(t[1])

    print(dict_words)
    print(dict_labels)
    print(longest_line)
    print(corpus)
    print(prediction)

    matrix_words = [[0 for x in range(longest_line)] for y in range(len(corpus))]
    matrix_labels = [[[0] for x in range(longest_line)] for y in range(len(corpus))]
    for i in range(len(corpus)):
        for j in range(longest_line):
            if len(corpus[i]) > j:
                if corpus[i][j] in dict_words:
                    matrix_words[i][j] = dict_words[corpus[i][j]]
                else:
                    matrix_words[i][j] = dict_words['oov']
            else:
                matrix_words[i][j] = dict_words['pad']
    for i in range(len(prediction)):
        for j in range(longest_line):
            if len(prediction[i]) > j:
                matrix_labels[i][j] = [dict_labels[prediction[i][j]]]
            else:
                matrix_labels[i][j] = [dict_labels['O']]

    X = np.asarray(matrix_words)
    Y = np.asarray(matrix_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    print(X_train.shape)
    print(y_train.shape)

    entree = Input(shape=(longest_line,),dtype='float32')
    emb = Embedding(len(dict_words), 100)(entree)
    bi = Bidirectional(CuDNNLSTM(20, return_sequences=True))(emb)
    # drop = Dropout(0.5)(bi)
    out = TimeDistributed(Dense(units=len(dict_labels),activation='tanh'))(bi)

    model = Model(inputs=entree,outputs=out)
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train,batch_size=5)

    # score = model.evaluate(X_test, y_test)

    # print(score)

    y_predict = model.predict(X_test)
    print(y_predict)

