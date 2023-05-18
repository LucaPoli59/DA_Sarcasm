# import silence_tensorflow.auto
import string

import pandas as pd
from random import randint
import numpy as np
import tensorflow
from nltk import LancasterStemmer, TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import metrics, losses, layers, activations, models, callbacks, utils, initializers
from keras.preprocessing.text import one_hot
from keras.datasets.imdb import load_data
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import timeit

TARGET = "sarcastic"
EPOCHS = 5
BATCH_SIZE = 128
ALT_DIR = "F:/programmazione/data_analytics_project/"


# MAX_TOKENS = 50000  # Dopo alcuni test ho visto che tale numero è più che sufficiente, lo si può calcolare dal value_count


def values_count_from_list(series, normalize=False):
    """
    Funzione che prende una serie che contiene liste, e restituisce gli elementi più comuni
    :param series: serie d'input
    :type series: pd.Series
    :param normalize: parametro per attivare la normalizzazione
    :type normalize: bool
    :return: Serie di value count
    :rtype: pd.Series
    """

    series_exploded = series.explode()
    return round(series_exploded.value_counts(normalize=normalize), 5)


def custom_split(input_str):
    return tensorflow.strings.split(input_str, sep=" <> ")


def create_embedding_matrix(glove_path, vocab, embedding_dim):
    """
    Funzione che crea la embedding matrix a <embedding_dim> usando un GloVe Pretrained Embedding
    :param glove_path: path al glove file
    :type glove_path: str
    :param vocab: dizionario contenente i termini usati
    :type vocab: dict
    :param embedding_dim: dimensione di embedding (deve essere compatibile con il file)
    :type embedding_dim: int
    :return: embedding matrix risultante
    :rtype: np.matrix
    """
    stemmer = LancasterStemmer()
    word_index = dict(zip(vocab, range(len(vocab))))
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    with open(glove_path, encoding="utf8") as glove_f:
        for line in glove_f:
            word, *vector = line.split()
            word_stemmed = stemmer.stem(word)
            if word_stemmed in word_index.keys():
                embedding_matrix[word_index[word_stemmed]] = np.array(vector, dtype='float32')

    return embedding_matrix


def parent_text_processing(parent_text, outlier_punctuation):

    tweet_tokenizer = TweetTokenizer()  # Tokenization
    tokenized = parent_text.apply(lambda x: tweet_tokenizer.tokenize(x))

    punctuation = list(string.punctuation)
    punctuation.append("...")
    del_punctuation = [point for point in punctuation if point not in outlier_punctuation]

    # eliminazione della punteggiatura
    tokenized = tokenized.apply(lambda word_list: [word for word in word_list if word not in del_punctuation])

    # eliminazione delle stopwords
    nsw = tokenized.apply(lambda word_list: [word for word in word_list if word not in stopwords.words('english')])

    stemmer = LancasterStemmer()
    nsw_st = nsw.apply(lambda word_list: [stemmer.stem(word) for word in word_list])

    return nsw_st


df = pd.read_json("dataset/train-processed.json", encoding='utf-8').sample(frac=0.33)
df['parent'] = parent_text_processing(df['parent'])
print(df['parent'])

random_state = randint(0, 1000)
test_size = 0.1

target_train, target_val = train_test_split(df[TARGET], random_state=random_state, test_size=test_size)
contex_train, contex_val = train_test_split(df[['author', 'subreddit', 'parent']],
                                            test_size=test_size, random_state=random_state)
# calcolo la len per il parent
parent_s_len = contex_train['parent'].apply(len)
q1, q3 = parent_s_len.quantile(0.25), parent_s_len.quantile(0.75)
parent_len = round(2 * (q3 + 1.5 * (q3 - q1)))

# calcolo len per gli altri
author_len = np.max(contex_train['author'].str.count(" ") + 1)
subreddit_len = np.max(contex_train['subreddit'].str.count(" ") + 1)

print("Parent len:\t", parent_len)
print("Author len:\t", author_len)
print("Subreddit len:\t", subreddit_len)

# Modello del parent

contex_train['parent'] = contex_train['parent'].apply(lambda words_list: " <> ".join(words_list))
contex_val['parent'] = contex_val['parent'].apply(lambda words_list: " <> ".join(words_list))

parent_vectorize_layer = layers.TextVectorization(
    max_tokens=None,
    standardize=None,
    split=custom_split,
    output_mode='int',
    output_sequence_length=parent_len,
    name="vectorizer"
)
parent_vectorize_layer.adapt(contex_train['parent'])
parent_vocabulary = parent_vectorize_layer.get_vocabulary()
embedding_matrix_parent = create_embedding_matrix(ALT_DIR + "glove/glove.42B.300d.txt", parent_vocabulary, 300)

score = np.count_nonzero(np.count_nonzero(embedding_matrix_parent, axis=1)) / len(parent_vocabulary)
print("L'Embedding matrix con GloVe è completa al ", round(score, 2) * 100, "%")

model_parent = models.Sequential(name="parent_processor")
model_parent.add(layers.Input(shape=(1,), name="input", dtype=tensorflow.string))
model_parent.add(parent_vectorize_layer)
model_parent.add(layers.Embedding(input_dim=len(parent_vocabulary),
                                  output_dim=300,
                                  input_length=parent_len,
                                  mask_zero=True,
                                  embeddings_initializer=initializers.initializers.Constant(embedding_matrix_parent),
                                  trainable=False))

model_parent.add(
    layer=layers.Bidirectional(layers.LSTM(50, return_sequences=False, name="lstm_bidirectional")))

model_parent.add(layers.Dense(1, activation=activations.sigmoid, name="output"))

model_parent.compile(optimizer="adam", loss=losses.BinaryCrossentropy(),
                     metrics=[metrics.BinaryAccuracy()])

model_parent.summary()

model_parent.fit(x=contex_train['parent'], y=target_train, validation_data=(contex_val['parent'], target_val),
                 batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# author
author_vectorize_layer = layers.TextVectorization(
    max_tokens=10000,
    standardize=None,
    split=None,
    output_mode='multi_hot',
    name="vectorizer"
).adapt(contex_train['author'])

model_author = models.Sequential(name="author_processor")
model_author.add(layers.Input(shape=(1,), name="input", dtype=tensorflow.string))
model_author.add(author_vectorize_layer)
model_author.add(layers.Dense(100, name="dense1", activation=activations.relu))
model_author.add(layers.Dense(1, activation=activations.sigmoid, name="output"))

model_author.compile(optimizer="adam", loss=losses.BinaryCrossentropy(),
                     metrics=[metrics.BinaryAccuracy()])

model_author.summary()

model_author.fit(x=contex_train['author'], y=target_train, validation_data=(contex_val['author'], target_val),
                 batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# subreddit
subreddit_vectorize_layer = layers.TextVectorization(
    max_tokens=1000,
    standardize=None,
    split=None,
    output_mode='multi_hot',
    name="vectorizer"
).adapt(contex_train['subreddit'])

model_subreddit = models.Sequential(name="subreddit_processor")
model_subreddit.add(layers.Input(shape=(1,), name="input", dtype=tensorflow.string))
model_subreddit.add(subreddit_vectorize_layer)
model_subreddit.add(layers.Dense(50, name="dense1", activation=activations.relu))
model_subreddit.add(layers.Dense(1, activation=activations.sigmoid, name="output"))

model_subreddit.compile(optimizer="adam", loss=losses.BinaryCrossentropy(),
                        metrics=[metrics.BinaryAccuracy()])

model_subreddit.summary()

model_subreddit.fit(x=contex_train['author'], y=target_train, validation_data=(contex_val['author'], target_val),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
