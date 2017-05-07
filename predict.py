#!/usr/bin/python

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense, LSTM, Dropout

from sklearn.preprocessing import MinMaxScaler

NB_EPOCH = 400

BATCH_SIZE = 64

LOOK_AHEAD = 30

PRINT_LOSS = False

PUSH_PREDICTIONS = False

def load_dataset(filename):
    scaler = MinMaxScaler(feature_range=(0, 1))

    df = pd.read_csv(filename, sep=';', usecols=[2])
    raw_seq = df.values.astype('float32')
    raw_seq = scaler.fit_transform(raw_seq)

    # truncate sequence to make it a multiple of BATCH_SIZE in length
    seq_len = len(raw_seq) - len(raw_seq) % BATCH_SIZE
    raw_seq = raw_seq[:seq_len]

    X = np.zeros((raw_seq.shape[0], 1, 1))
    for i, x in enumerate(raw_seq):
        X[i, 0, 0] = x

    y = np.zeros((len(X), 1))
    for i in range(len(X) - LOOK_AHEAD):
        y[i, 0] = np.mean(X[i + 1: i + LOOK_AHEAD])
        #y[i, 0] = X[i + LOOK_AHEAD]

    return X, y

class ResetStatesCallback(Callback):
    def on_epoch_end(self, batch, logs={}):
        self.model.reset_states()

def main(args=sys.argv):
    if len(args) != 2:
        sys.exit('Usage: %s <filename>', args[0])

    X, y = load_dataset(args[1])

    bis = (BATCH_SIZE, X.shape[1], X.shape[2])

    model = Sequential()
    model.add(LSTM(64, batch_input_shape=bis, return_sequences=True, stateful=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False, stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='rmsprop')

    hist = model.fit(X, y,
                     callbacks=[ResetStatesCallback()],
                     shuffle=False,
                     nb_epoch=NB_EPOCH,
                     batch_size=BATCH_SIZE,
                     verbose=1)

    if PRINT_LOSS:
        plt.plot(hist.history['loss'])
        plt.show()

    predicted = model.predict(X, batch_size=BATCH_SIZE)

    # Let the network produce results in unseen areas
    if PUSH_PREDICTIONS:
        predicted = np.expand_dims(predicted, axis=1)
        predicted = model.predict(predicted, batch_size=BATCH_SIZE)

    plt.plot(y)
    plt.plot(predicted)
    plt.show()

if __name__ == '__main__':
    main()
