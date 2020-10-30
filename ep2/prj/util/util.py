# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from nltk.tokenize import WordPunctTokenizer
# from nltk.corpus import stopwords as nltk_stopwords
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

import os
import re

tknzr_WordPunctTokenizer = WordPunctTokenizer()
# stopwords = nltk_stopwords.words('portuguese')


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def train_test_val_split(dataset, train_size=0.6, test_size=0.3, colname_stratify='overall_rating', random_seed=29):
    val_size = 1 - round((train_size + test_size),1)
    split_train_test_size = test_size + val_size

    train, val = train_test_split(dataset, 
                                  test_size=split_train_test_size, 
                                  stratify=dataset[colname_stratify], 
                                  random_state=random_seed)

    test, val = train_test_split(val, 
                                  test_size=val_size/split_train_test_size, 
                                  stratify=val[colname_stratify], 
                                  random_state=random_seed)
    return train.reset_index(drop=True), test, val

def getXY(serieX, serieY, padding_maxlen=50):
#     x_train = keras.preprocessing.sequence.pad_sequences(train['review_text_clean'], maxlen=padding_maxlen, padding='post')
    x_ = serieX.values
    y_ = serieY.values
    return x_, y_

def prepare_text(x):
    tkn = tknzr_WordPunctTokenizer.tokenize(x.lower())
    tkn = [  re.sub(r'(.)\1+', r'\1', w )  for w in tkn ]
    # tkn = [  re.sub(r'(.)\1+', r'\1', w )  for w in tkn 
    # 				if w not in stopwords and w.isalpha() and len(w) < 14 ]
#     and w.isalpha()
#     if len(tkn) > 70:
#         tkn = ' '.join( tkn[0:35] + ['#'] + tkn[-35:] )
#     else:
    tkn = ' '.join( tkn )
    return tkn

def sort_by_size(df, col_to_sort):
    df['sentence_length'] = df[col_to_sort].apply(lambda x: len(x.split()))
    df.sort_values(by=['sentence_length'], inplace=True, ignore_index=True)
    return df


def preprocess(df_to_work):

    values_to_retain=[1,2,3,4,5]
    df_to_work = df_to_work[df_to_work['overall_rating'].isin(values_to_retain)]
    # df_to_work
    df_to_work['overall_rating'] = df_to_work.overall_rating.apply(lambda x: x-1)


    # df_to_work['review_text'] = df_to_work.review_text.apply(lambda x: ' '.join( tknzr_WordPunctTokenizer.tokenize(x.lower()) ))
    df_to_work['review_text'] = df_to_work.review_text.apply(lambda x: prepare_text(x) )

    return df_to_work

def get_vectorize_layer_adapted( data_to_adpat, vocab_size, seq_len, p_standardize=None):
    vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=vocab_size,
        standardize=p_standardize, # 'lower_and_strip_punctuation', # None
        output_mode='int', 
        output_sequence_length= seq_len,  # Only valid in INT mode.
        name="TextVec_%d" % vocab_size
    )
    vectorize_layer.adapt(data_to_adpat)
    return vectorize_layer


def get_nilc_embedding_layer(path_to_nilc, vectorize_layer):
    print("[Preparando camada do NILC...]")

    # https://keras.io/examples/nlp/pretrained_word_embeddings/#create-a-vocabulary-index
    # Carregar o Word2Vec do NILC
    print("[Lendo o NILC de..:", path_to_nilc, "]")
    model_w2v = KeyedVectors.load_word2vec_format(path_to_nilc)

    voc = vectorize_layer.get_vocabulary()
    word_index = dict(zip(voc, range(2,len(voc))))

    num_tokens = len(voc) + 2
    embedding_dim = 50
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        if word.decode('UTF-8') in model_w2v.vocab:
            embedding_matrix[i] = model_w2v[ word.decode('UTF-8') ]
            hits += 1
        else:
            misses += 1

    print("Converted %d words (%d misses)" % (hits, misses))

    nilc_embedding_layer = layers.Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
        name="Emb_NILC_%d_tokens_%d_miss" % (num_tokens, misses)
    )
    return nilc_embedding_layer



def create_model(vectorize_layer, embed_layer, is_birect=False, vocab_size=1000, dropout_prob=0.0, lstm_units=32):

    lstm_layer = None
    if is_birect:
        # Camada Bidirecional
        lstm_layer = layers.Bidirectional(
            layers.LSTM(units=lstm_units, activation="tanh", recurrent_activation="sigmoid"),
            backward_layer=layers.LSTM(units=lstm_units, activation="relu", recurrent_activation="sigmoid", go_backwards=True),
            name="BIDIRECT_LSTM_%d" % lstm_units    
            )
    else:
        lstm_layer = layers.LSTM(
                        units=lstm_units,
                        activation="tanh",
                        recurrent_activation="sigmoid",
                        name="LSTM_%d" % lstm_units)

    model = keras.Sequential()
    model.add(layers.Input(shape=(1,), dtype=tf.string, name="Input_shape1"))
    model.add(vectorize_layer)

    # model.add(nilc_embedding_layer)
    model.add(embed_layer)

    model.add(lstm_layer)
    # model.add(
    #     layers.LSTM(
    #         units=lstm_units,
    #         activation="tanh",
    #         recurrent_activation="sigmoid",
    #         name="LSTM_%d" % lstm_units
    #     )
    # )
    model.add(layers.Dropout(dropout_prob, name=f"Dropout_{dropout_prob}"))
    model.add(keras.layers.Dense(5, activation='softmax', name="Dense_5"))
    
    return model


def renderfig( file, line1, line2, label1, label2, title, xlabel, ylabel, text ):
    epochs = range(1, len(line1) + 1)
    plt.plot(epochs, line1, 'r', label=label1)
    plt.plot(epochs, line2, 'b', label=label2)
    plt.title(title)
    plt.xlabel(xlabel )
    plt.ylabel(ylabel)
    plt.legend()
    plt.figtext(0.2, -0.34, text,  fontsize=14)
    plt.savefig( file, bbox_inches = "tight")
    plt.close('all')
