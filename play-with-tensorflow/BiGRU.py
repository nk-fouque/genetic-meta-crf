#!/usr/bin/python3
import subprocess
import sys
from collections import namedtuple

import numpy as np
from keras import Input, Model
from keras.layers import Bidirectional, Embedding, TimeDistributed
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from sklearn.model_selection import train_test_split
import tensorflow as tf

# import tensorflow_addons as tfa
# configT = tf.ConfigProto()
# configT.gpu_options.allow_growth = True
# session = tf.Session(config=configT)


if __name__ == '__main__':
    args = sys.argv
    DATASET = args[1]
    dict_words = dict()
    dict_words['pad'] = 0
    dict_words['oov'] = 1

    dict_labels = dict()
    dict_labels['pad'] = 0

    longest_line = 86

    with open(DATASET, "r") as f:
        corpus = []
        prediction = []
        corpus_current = []
        prediction_current = []
        for line in f:
            t = line.split()
            if len(t) == 0:
                corpus.append(corpus_current)
                corpus_current = []
                prediction.append(prediction_current)
                prediction_current = []
                continue
            if (t[0] not in dict_words):
                dict_words[t[0]] = len(dict_words)
            if (t[1] not in dict_labels):
                dict_labels[t[1]] = len(dict_labels)
            corpus_current.append(t[0])
            prediction_current.append(t[1])

    # print(dict_words)
    # print(dict_labels)
    # print(longest_line)
    # print(corpus)
    # print(prediction)

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
                matrix_labels[i][j] = [dict_labels['pad']]

    X = np.asarray(matrix_words)
    Y = np.asarray(matrix_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # print(X_train.shape)
    # print(y_train.shape)

    entree = Input(shape=(longest_line,), dtype='int32')
    emb = Embedding(len(dict_words), 100)(entree)
    bi = Bidirectional(GRU(120, use_bias=True,return_sequences=True))(emb)
    drop = Dropout(0.4)(bi)
    out = TimeDistributed(Dense(units=len(dict_labels), activation='softmax'))(drop)

    model = Model(inputs=entree, outputs=out)
    model.summary()

    # print(y_predict)

    dict_labels_revert = {v: k for k, v in dict_labels.items()}
    dict_words_revert = {v: k for k, v in dict_words.items()}

    # print(predictions_list)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=[tf.metrics.SparseCategoricalAccuracy()])

    model.fit(X_train, y_train, batch_size=128, epochs=60)

    # score = model.evaluate(X_test, y_test)

    # print(score)

    y_predict = model.predict(X_test)

    predictions_list = []
    for line, origin_labels in zip(y_predict, y_test):
        line_list = []
        for label_num, true_label in zip(line, origin_labels):
            line_list.append((dict_labels_revert[np.argmax(label_num)], dict_labels_revert[true_label[0]]))
        predictions_list.append(line_list)

    LABEL_FILE = 'generated_files/red_CuDDNN'
    EVAL_FILE = "generated_files/evalLSTM"

    with open(LABEL_FILE, 'w+') as f:
        for line in predictions_list:
            for word in line:
                if word[1] != 'pad':
                    f.write('dummy\t' + word[1] + '\t' + word[0] + '\n')
            f.write('\n')

    try:
        subprocess.check_call('cat ' + LABEL_FILE + ' | perl evaluation.pl > ' + EVAL_FILE, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.cmd)
        print(e.output)

    Evaluation = namedtuple("Evaluation", "accuracy precision recall f1")
    with open(EVAL_FILE, "r") as f:
        for i, line in enumerate(f):
            if (i == 1):
                l = line.split()
                accuracy = l[1]
                precision = l[3]
                recall = l[5]
                f1 = l[7]
                print(Evaluation(accuracy, precision, recall, f1))
                break
