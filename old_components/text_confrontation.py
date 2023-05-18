import timeit

import numpy as np
import pandas as pd
from random import randint

import tensorflow
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import metrics, losses, layers, activations, models, callbacks, utils
from keras.preprocessing.text import one_hot
from keras.datasets.imdb import load_data
from keras.utils import pad_sequences
from scipy.stats import zscore

TARGET = "sarcastic"


def compute_helpful_words(text, target, vocabulary_size=1000, precision=3):
    """
    Funzione che calcola le parole più utili per un modello, calcolando le proporzioni in cui compaiono sarcastiche e
    non, per poi mantenere quelle che superano lo z score
    :param text: serie contente il testo da elaborare (sotto forma di lista di token)
    :type text: pd.Series
    :param target: serie del target che (avendo lo stesso indice di text) fornisce la label
    :type target: pd.Series
    :param vocabulary_size: numero di feature-parole da estrarre
    :type vocabulary_size: int or None
    :param precision: precisione di esclusione
    :type precision: float
    :return: dataframe che ha come indice le parole, e come valori le loro proporzioni
    :rtype: pd.DataFrame
    """
    text_vectorizer = CountVectorizer(max_features=vocabulary_size)
    text = text.apply(lambda words_list: " ".join(words_list))

    text_vectorized = text_vectorizer.fit_transform(text)
    target = target.to_frame(TARGET)
    target['sparse_index'] = np.arange(len(target))

    index_s = target.loc[target[TARGET] == 1, 'sparse_index']
    index_ns = target.loc[target[TARGET] == 0, 'sparse_index']
    text_vectorized_s, text_vectorized_ns = text_vectorized[index_s.values] > 0, text_vectorized[index_ns.values] > 0

    text_s_prop = pd.DataFrame(index=text_vectorizer.get_feature_names_out(), columns=['sarcastic', 'not_sarcastic'])
    text_s_prop['sarcastic'] = np.array(text_vectorized_s.sum(axis=0))[0] / len(index_s)
    text_s_prop['not_sarcastic'] = np.array(text_vectorized_ns.sum(axis=0))[0] / len(index_ns)
    print(text_s_prop)
    text_s_prop['rate'] = text_s_prop['sarcastic'] / (text_s_prop['sarcastic'] + text_s_prop['not_sarcastic'])
    text_s_prop['tot_occ'] = (text_s_prop['sarcastic'] * len(index_s) +
                              text_s_prop['not_sarcastic'] * len(index_ns))

    # elimino le parole che occorrono poco
    text_s_prop = text_s_prop.loc[text_s_prop['tot_occ'] > text_s_prop['tot_occ'].quantile(0.3)]

    return text_s_prop.loc[abs(zscore(text_s_prop['rate'])) >= precision].sort_values(by='rate', ascending=False)


TARGET = "sarcastic"

df = pd.read_json("dataset/train-processed.json", encoding='utf-8')

random_state = randint(0, 1000000)
test_size = 0.1

target_train, target_val = train_test_split(df[TARGET], random_state=random_state, test_size=test_size)
nsw_train, nsw_val = train_test_split(df['text_nsw'], test_size=test_size, random_state=random_state)
nsw_st_train, nsw_st_val = train_test_split(df['text_nsw_st'], test_size=test_size, random_state=random_state)
st_train, st_val = train_test_split(df['text_st'], test_size=test_size, random_state=random_state)

precision = 2
vocab_size = None

nsw_hw = compute_helpful_words(nsw_train, target_train, vocabulary_size=vocab_size, precision=precision)
nsw_st_hw = compute_helpful_words(nsw_st_train, target_train, vocabulary_size=vocab_size, precision=precision)
st_hw = compute_helpful_words(st_train, target_train, vocabulary_size=vocab_size, precision=precision)

print("nsw:\n", nsw_hw, "\n\nnsw_st:\n", nsw_st_hw, "\n\nst:\n", st_hw)
print("\n\nIl migliore è:\t",
      pd.Series(index=['nsw', 'nsw_st', 'st'], data=[len(text) for text in [nsw_hw, nsw_st_hw, st_hw]]
                ).sort_values(ascending=False))

