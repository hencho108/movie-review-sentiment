import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, Embedding
from tensorflow.keras.models import Model


def run_training():
    """Train the model"""

    # load data
    data = pd.read_csv("../training/data/IMDB Dataset.csv")

    print('Preprocessing data...')
    # create binary labels
    data['label'] = data['sentiment'].map({'negative': 0, 'positive': 1})
    Y = data['label'].values

    # Remove HTML tags
    data['review'] = [BeautifulSoup(text, "html.parser").get_text() for text in data['review']]

    # split up the data
    df_train, df_test, Y_train, Y_test = train_test_split(data['review'], Y, test_size=0.33)

    # Convert sentences to sequences
    MAX_VOCAB_SIZE = 10000
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(df_train)
    sequences_train = tokenizer.texts_to_sequences(df_train)
    sequences_test = tokenizer.texts_to_sequences(df_test)

    # get word -> integer mapping
    word2idx = tokenizer.word_index

    # pad sequences so that we get a N x T matrix
    data_train = pad_sequences(sequences_train, maxlen=500)
    data_test = pad_sequences(sequences_test, maxlen=500)

    ### Create the model
    print('Building model...')
    V = len(word2idx) # number of unique tokens
    T = data_train.shape[1] # sequence length
    D = 32 # Embedding dimensionality
    M = 10 # Hidden state dimensionality

    i = Input(shape=(T,))
    x = Embedding(V + 1, D)(i)
    x = LSTM(M, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation='sigmoid')(x)

    model = Model(i, x)

    # Compile and fit
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('Training model...')
    model.fit(data_train,Y_train,epochs=1,validation_split=0.3)

    # Final evaluation of the model
    scores = model.evaluate(data_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    model.save('rnn_model.h5')


if __name__ == '__main__':
    run_training()
