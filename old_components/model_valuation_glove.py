# import silence_tensorflow.auto
import pandas as pd
from random import randint

import tensorflow
from nltk import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import metrics, losses, layers, activations, models, callbacks, utils, initializers
import numpy as np


TARGET = "sarcastic"
EPOCHS = 10
BATCH_SIZE = 128
ALT_DIR = "F:/programmazione/data_analytics_project/"


def custom_split(input_str):
    return tensorflow.strings.split(input_str, sep=" <> ")


def create_embedding_matrix(glove_path, vocab, embedding_dim, stemming=True):
    """
    Funzione che crea la embedding matrix a <embedding_dim> usando un GloVe Pretrained Embedding
    :param glove_path: path al glove file
    :type glove_path: str
    :param vocab: dizionario contenente i termini usati
    :type vocab: dict
    :param embedding_dim: dimensione di embedding (deve essere compatibile con il file)
    :type embedding_dim: int
    :param stemming: indica se effettuare lo stemming delle parole
    :type stemming: bool
    :return: embedding matrix risultante
    :rtype: np.matrix
    """
    stemmer = LancasterStemmer()
    word_index = dict(zip(vocab, range(len(vocab))))
    embedding_matrix = np.zeros((len(vocab), embedding_dim))

    with open(glove_path, encoding="utf8") as glove_f:
        for line in glove_f:
            word, *vector = line.split()
            if stemming:
                word = stemmer.stem(word)
            if word in word_index.keys():
                embedding_matrix[word_index[word]] = np.array(vector, dtype='float32')

    return embedding_matrix


def evaluate_model(text_train, y_train, text_val, y_val, sentence_max_len, model_name, stemming=True):
    # dobbiamo riunire i token per poterli dividere poi nel vectorizer
    text_train = text_train.apply(lambda words_list: " <> ".join(words_list))
    text_val = text_val.apply(lambda words_list: " <> ".join(words_list))

    vectorize_layer = layers.TextVectorization(
        max_tokens=None,
        standardize=None,
        split=custom_split,
        output_mode='int',
        output_sequence_length=sentence_max_len,
        name="vectorizer"
    )

    vectorize_layer.adapt(text_train)
    vocab = vectorize_layer.get_vocabulary()

    embedding_matrix = create_embedding_matrix(ALT_DIR + "glove/glove.42B.300d.txt", vocab, 300, stemming=stemming)

    model = models.Sequential(name=model_name)
    model.add(layers.Input(shape=(1,), name="input", dtype=tensorflow.string))
    model.add(vectorize_layer)
    model.add(layers.Embedding(input_dim=len(vocab),
                               output_dim=300,
                               input_length=sentence_max_len,
                               mask_zero=True,
                               embeddings_initializer=initializers.initializers.Constant(embedding_matrix),
                               trainable=False))

    model.add(
        layer=layers.Bidirectional(layers.LSTM(50, return_sequences=True, name="lstm_bidirectional")))
    model.add(
        layer=layers.Bidirectional(layers.LSTM(10, return_sequences=False, name="lstm_bidirectional2")))

    model.add(layers.Dense(1, activation=activations.sigmoid, name="output"))

    model.compile(optimizer="adam", loss=losses.BinaryCrossentropy(),
                  metrics=[metrics.BinaryAccuracy()])

    model.summary()
    print("Fitting del modello ", model_name, ":\n")

    model.fit(x=text_train, y=y_train, validation_data=(text_val, y_val), batch_size=BATCH_SIZE,
              epochs=EPOCHS, verbose=1, callbacks=[callbacks.TensorBoard(log_dir=str(ALT_DIR + "ris/logs/" + model_name)
                                                                         , write_images=True)])
    # model.save(str(SAVE_DIR + "model/" + model_name), save_format="tf")
    utils.plot_model(model, to_file="ris/model_eval.png", show_shapes=True)

    return model


# N.B. Qua uso il tokenizer sulle frasi, ma è meglio usarlo sui token risultati dall'altra fase
df = pd.read_json("dataset/train-processed.json", encoding='utf-8')

random_state = randint(0, 1000)
test_size = 0.1

target_train, target_val = train_test_split(df[TARGET], random_state=random_state, test_size=test_size)
nsw_train, nsw_val = train_test_split(df['text_nsw'], test_size=test_size, random_state=random_state)
nsw_st_train, nsw_st_val = train_test_split(df['text_nsw_st'], test_size=test_size, random_state=random_state)
st_train, st_val = train_test_split(df['text_st'], test_size=test_size, random_state=random_state)

print("Line number:\t", len(target_train))

max_len = dict.fromkeys([str(nsw_train.name), str(nsw_st_train.name), str(st_train.name)])
for serie in [nsw_train, nsw_st_train, st_train]:
    serie_len = serie.apply(len)
    plt.subplots()
    serie_len.plot.box(title=str("Distribuzione con outliers di " + str(serie.name)))
    plt.subplots()
    serie_len.plot.box(title=str("Distribuzione senza outliers di " + str(serie.name)), showfliers=False)

    q1, q3 = serie_len.quantile(0.25), serie_len.quantile(0.75)
    limit = q3 + 1.5 * (q3 - q1)
    print("Proporzione di outliers di", str(serie.name), ":\t", (serie_len > limit).sum() * 100 / len(serie_len))
    print("Proporzione di 2outliers di", str(serie.name), ":\t", (serie_len > 2 * limit).sum() * 100 / len(serie_len))
    print("Quantile candidato per ", str(serie.name), ":\t", limit, "\n")
    max_len[str(serie.name)] = round(2 * limit)

# plt.show()

eval_model_nsw = evaluate_model(nsw_train, target_train, nsw_val, target_val, max_len[str(nsw_train.name)], "nsw",
                                stemming=False)
eval_model_nsw_st = evaluate_model(nsw_st_train, target_train, nsw_st_val, target_val, max_len[str(nsw_st_train.name)],
                                   "nsw_st")
eval_model_st = evaluate_model(st_train, target_train, st_val, target_val, max_len[str(st_train.name)], "st")

eval_model_nsw.summary()

# Siccome è necessario avere sequenze di pari lunghezze studiamo la distribuzione delle lunghezze
