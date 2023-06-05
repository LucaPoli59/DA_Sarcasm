# -*- coding: utf-8 -*-
"""Data_Analytics_Finale_TMP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eHZZNzoqZB546ro1nZCEeBDtsWPKP3KH

## Progetto per Data Analytics: Sarcasm Detection

### Preparazione dell'ambiente di runtime

Import delle librerie
"""

import re
import string
import pandas as pd
from random import randint
import tensorflow
import nltk
import keras_tuner
from nltk import LancasterStemmer, TweetTokenizer
from nltk.corpus import stopwords
from scipy.stats import zscore
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import metrics, losses, layers, activations, models, callbacks, utils, initializers, Input
import numpy as np
from wordcloud import WordCloud
from plotly.express.colors import sample_colorscale
import os
import constants
import json



"""Dowload dei file per nltk"""

# nltk.download('stopwords')


"""Definizione di alcune costanti"""



"""### Fase di import del dataset e prima analisi
In questa fase verrà importato il dataset (suddividendolo in train e validation set) e si analizzerà:
- Il numero di righe, la presenza di duplicate e di nulle
- La distribuzione del target
- La presenza di elementi del contesto ripetuti

"""

df_full = pd.read_csv(os.path.join(constants.DATA_IN_PATH, "data_full.tsv"),
                      sep="\t", names=[constants.TARGET, "text", "author", "subreddit", "date", "parent"]).sample(frac=0.05)

df_full.to_csv(os.path.join(constants.DATA_OUT_PATH, "data_full_sample.csv"))

df_train, df_val = train_test_split(df_full, test_size=0.1)

if constants.ENABLE_OUT:
    print("Dimensione del dataset:\t", len(df_train), "\n")
    print("tipi di variabile:\n", df_train.dtypes, "\n")
    print("Prime righe di esempio:\n", df_train.head(3), "\n")
    print("Righe nulle:\n", df_train.isna().sum(), "\n")
    print("Righe duplicate:\n", df_train.duplicated().sum(), "\n")

"""Si definisce un metodo di preprocessing che:
- Rimuova le righe duplicate
- Rimuova le righe nulle (visto che sono poche non è necessario imputarle)
- Converta i tipi di variabile
"""


def dataset_opening_preprocessing(dataframe):
    """
    Funzione generata dalla prima analisi del dataframe, serve per eliminare righe nulle e duplicate e
    per effettuare le conversioni di tipo
    :param dataframe: dataframe d'input
    :type dataframe: pd.DataFrame
    :return: dataframe processato
    :rtype: pd.DataFrame
    """
    dataframe = dataframe.dropna().drop_duplicates().reset_index(drop=True)
    dataframe.index.name = "index"

    dataframe[constants.TARGET] = dataframe[constants.TARGET].astype("bool")
    str_var = ['text', 'parent', 'subreddit', 'author']
    dataframe[str_var] = dataframe[str_var].astype("str")
    dataframe['date'] = pd.to_datetime(dataframe['date'], format="%Y-%m")
    dataframe['text'], dataframe['parent'] = dataframe['text'].str.lower(), dataframe['parent'].str.lower()
    return dataframe


df_train = dataset_opening_preprocessing(df_train)
df_train.to_csv(os.path.join(constants.DATA_OUT_PATH, "train.csv"))

target_info_rate = df_train['sarcastic'].value_counts(normalize=True).max()

if constants.ENABLE_OUT:
    print("tipi di variabile dopo la conversione:\n", df_train.dtypes, "\n")
    # Analisi del target
    print("Stampa di 3 righe sarcastiche:\n",
          df_train.loc[df_train[constants.TARGET] == 1].head(3)[[constants.TARGET, 'text']], "\n")
    print("Stampa di 3 righe non sarcastiche:\n",
          df_train.loc[df_train[constants.TARGET] == 0].head(3)[[constants.TARGET, 'text']], "\n\n")
    print("Distribuzione del target:", round(target_info_rate * 100, 2))


def sarcastic_proportion_count(df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
    """
    Calcola la proporzione di sarcastic e non sarcastic delle feature uniche della colonna feature
    :param df: dataset contenente il target sarcastic e la feature
    :param target_rate: proporzione di sarcastic nel dataset
    :return: dataframe contenete per ogni feature unica, la sua frequenza e proporzione
    """

    sc_rows = df[constants.TARGET] == 1
    sc_vc = df.loc[sc_rows, df.columns[1:]].value_counts()
    nsc_vc = df.loc[~sc_rows, df.columns[1:]].value_counts()

    df_freq = pd.DataFrame({'sarc_freq': sc_vc, 'n_sarc_freq': nsc_vc}).fillna(0)
    df_freq['tot'] = df_freq['sarc_freq'] + df_freq['n_sarc_freq']
    df_freq['prop'] = df_freq['sarc_freq'] / df_freq['tot']
    df_freq['info_rate'] = abs(df_freq['prop'] - target_rate) * 100

    if len(df_freq.index.names) == 1:
        df_freq.index = df_freq.index.get_level_values(0)
        df_freq.index.name = "element"

    df_freq = df_freq.sort_values(by='tot', ascending=False)[['tot', 'prop', 'sarc_freq', 'n_sarc_freq', 'info_rate']]
    return df_freq


for context_feature in ['subreddit', 'author', 'date', 'parent']:
    sarc_prop = sarcastic_proportion_count(df_train[[constants.TARGET, context_feature]], target_info_rate)
    sarc_prop.to_csv(os.path.join(constants.DATA_SP_PATH, context_feature + ".csv"))

    if constants.ENABLE_OUT:
        print("\nAnalisi delle proporzioni sarcastiche per ", context_feature, ":\n", sarc_prop.head(5), "\n\n")

df_train_len = df_train[['sarcastic', 'text', 'parent']].copy()
df_train_len[['text', 'parent']] = df_train_len[['text', 'parent']].applymap(lambda x: len(x.split()))



for feature in ['text', 'parent']:
    sarc_prop = sarcastic_proportion_count(df_train_len.loc[abs(zscore(df_train_len[feature])) < 3,
                                           [constants.TARGET, feature]], target_info_rate)
    sarc_prop.to_csv(os.path.join(constants.DATA_SP_PATH, "len_" + feature + ".csv"))

    if constants.ENABLE_OUT:
        print("\nAnalisi delle proporzioni sarcastiche per la lunghezza di ", feature, ":\n", sarc_prop.head(5), "\n\n")

sarc_prop = sarcastic_proportion_count(df_train_len.loc[(abs(zscore(df_train_len[['text', 'parent']])) < 3).all(axis=1),
                                                        [constants.TARGET, 'text', 'parent']], target_info_rate)
sarc_prop.to_csv(os.path.join(constants.DATA_SP_PATH, "len_text_parent.csv"))


"""## Fase di analisi del testo
In questa fase si analizza il testo del commento (quindi la feature 'text').
Verranno analizzati i token di cui esso si compone, e come la frequenza di essi varia nelle fasi di:
- Eliminazione della punteggiatura;
- Eliminazione delle stopwords;
- Stemming.

Producendo tre tipi di testo:
- nsw: senza stopwords;
- nsw_st: senza stopwords e con stemming;
- st: con stemming.
che verranno poi confrontati nella successiva analisi
"""


def word_cloud_generator(df_sp, save_name):
    """
    Funzione che genera la wordcloud di una serie di frequenza di parole,
    i cui colori sono in accordo con la color scale e dipendono dall'info_rate
    :param df_sp: dataframe contenente la sarcastic proportion di una feature
    :type df_sp: pd.DataFrame
    :param save_name: nome del file di salvataggio
    :type save_name: str
    :return: wordcloud generata
    :rtype: WordCloud
    """
    color_map = df_sp['info_rate'].to_frame().reset_index()
    # color_map = df_sp['info_rate'].to_frame().reset_index().sort_values(by='info_rate')
    info_min, info_max = color_map['info_rate'].min(), color_map['info_rate'].max()
    color_map['rate_s'] = (color_map['info_rate'] - info_min) / (info_max - info_min)
    color_map['color'] = sample_colorscale(constants.COLOR_SCALE, color_map['rate_s'])
    color_dict = color_map.set_index('element')['color']

    wc = WordCloud(width=1600, height=800, background_color='white',
                   color_func=lambda *args, **kwargs: color_dict[args[0]]
                   ).generate_from_frequencies(df_sp['tot'].to_dict())

    with open(os.path.join(constants.DATA_WC_PATH, save_name + ".json"), 'w') as json_file:
        json.dump(wc.to_array().tolist(), json_file)

    return wc


"""Si procede con la tokenizzazione del testo"""

tweet_tokenizer = TweetTokenizer()

df_train['text_tokenized'] = df_train['text'].apply(lambda x: tweet_tokenizer.tokenize(x))

if constants.ENABLE_OUT:
    print("stampa di tre frasi con i relativi token:\n", df_train[['text', 'text_tokenized']].head(3), "\n\n")

"""Prima di eliminare la punteggiatura, si esamina la frequenza dei simboli in frasi sarcastiche e non, in quanto essi possono essere fonte d'informazione."""

all_punctuation = list(string.punctuation)
all_punctuation.append("...")

punctuation_freq = pd.DataFrame(columns=["sarc_freq", "n_sarc_freq"], index=all_punctuation, dtype="float64")

for mark in punctuation_freq.index:
    text_wm = df_train.loc[df_train['text'].str.contains(re.escape(mark)), [constants.TARGET, 'text']]
    punctuation_freq.loc[mark, 'sarc_freq'] = (text_wm[constants.TARGET] == 1).sum()
    punctuation_freq.loc[mark, 'n_sarc_freq'] = (text_wm[constants.TARGET] == 0).sum()

punctuation_freq['tot'] = punctuation_freq['sarc_freq'] + punctuation_freq['n_sarc_freq']
punctuation_freq['prop'] = punctuation_freq['sarc_freq'] / punctuation_freq['tot']
punctuation_freq['info_rate'] = abs(punctuation_freq['prop'] - target_info_rate) * 100
punctuation_freq = punctuation_freq.dropna()
punctuation_freq = punctuation_freq.sort_values(by='tot', ascending=False)

punctuation_freq.index.name = "element"
punctuation_freq.to_csv(os.path.join(constants.DATA_SP_PATH, "text_punctuation.csv"))
word_cloud_generator(punctuation_freq, "text_punctuation")

if constants.ENABLE_OUT:
    print("Frequenza della punteggiatura in frasi sarcastiche:\n", punctuation_freq, "\n\n")
    plt.subplots()
    punctuation_freq['prop'].plot(kind='bar', title="Frequenza della punteggiatura in frasi sarcastiche")
    plt.show()


"""Vista la distribuzione del rateo dei simboli, si decide di mantenere quelli che sono degli outliers alla distribuzione (perchè possono discriminare meglio una frase sarcastica o non)"""

# Rimozione
outlier_punctuation = punctuation_freq.loc[abs(zscore(punctuation_freq['prop'])) >= 3].index.values
del_punctuation = [point for point in list(all_punctuation) if point not in outlier_punctuation]
df_train['text_tokenized'] = df_train['text_tokenized'].apply(
    lambda word_list: [word for word in word_list if word not in del_punctuation])


"""Si procede con l'eliminazione delle stopwords"""

df_train['text_nsw'] = df_train['text_tokenized'].apply(
    lambda word_list: [word for word in word_list if word not in stopwords.words('english')])


"""Si termina la fase con lo stemming"""

# Stemming
stemmer = LancasterStemmer()
df_train['text_nsw_st'] = df_train['text_nsw'].apply(
    lambda word_list: [stemmer.stem(word) for word in word_list])

# Stemming con stopwords
df_train['text_st'] = df_train['text_tokenized'].apply(lambda word_list: [stemmer.stem(word) for word in word_list])



"""## Confronto dei tipi di testo
In questa fase verranno confrontati i tre tipi di testi prodotti (nsw, nsw_st, st), in modo da individuare quale di essi porta più informazioni.

"""

"""## Confronto tramite sarcastic value rate
"""

for text_type in ['text_nsw', 'text_nsw_st', 'text_st', 'text_tokenized']:
    sarc_prop = sarcastic_proportion_count(df_train[[constants.TARGET, text_type]].explode(column=text_type),
                                           target_info_rate)
    sarc_prop.to_csv(os.path.join(constants.DATA_SP_PATH, text_type + ".csv"))
    word_cloud_generator(sarc_prop, text_type)

    if constants.ENABLE_OUT:
        print("\nAnalisi delle proporzioni sarcastiche per testo di tipo ", text_type, ":\n", sarc_prop.head(5), "\n\n")

train_text = df_train[['text_tokenized', 'text_nsw', 'text_nsw_st', 'text_st']].rename({
    'text_tokenized': 'tokenized', 'text_nsw': 'nsw', 'text_nsw_st': 'nsw_st', 'text_st': 'st'}, axis='columns')

train_text = train_text.applymap(lambda x: " ".join(x))
train_text.to_csv(os.path.join(constants.DATA_OUT_PATH, "train_text.csv"))

df_train = df_train.drop(columns=['text', 'text_tokenized', 'text_nsw_st', 'text_st']
                         ).rename(columns={'text_nsw': 'text'})

"""## Fase di preprocessing finale
In questa fase avviene il preprocessing finale prima del modello:
- Si elabora il testo parent come il testo del commento
- Si calcola la lunghezza delle frasi per il parent e per il testo
- Si preparano i dati del set di validation (usando i principi estratti dal train set)
"""


def text_processing(text, punctuation, join_str=None):
    """
    Funzione che riapplica gli step di processing del testo, secondo la pipeline già eseguita
    :param text: serie contenente il testo
    :type text: pd.Series
    :param punctuation: punti da eliminare
    :type punctuation: list
    :param join_str: stringa usata per effettuare l0unione dei token (riconvertendo il testo da lista a stringa)
    :type join_str: str
    :return: serie del testo rielaborata
    :rtype: pd.Series
    """

    tokenizer = TweetTokenizer()  # Tokenization
    tokenized = text.apply(lambda x: tokenizer.tokenize(x))

    # eliminazione della punteggiatura
    tokenized = tokenized.apply(lambda word_list: [word for word in word_list if word not in punctuation])

    # eliminazione delle stopwords
    nsw = tokenized.apply(lambda word_list: [word for word in word_list if word not in stopwords.words('english')])

    # stem = LancasterStemmer()
    # nsw_st = nsw.apply(lambda word_list: [stem.stem(word) for word in word_list])

    # if join_str is not None:
    #     nsw_st = nsw_st.apply(lambda words_list: join_str.join(words_list))
    #
    # return nsw_st
    if join_str is not None:
        nsw_st = nsw.apply(lambda words_list: join_str.join(words_list))

    return nsw


def dataset_processing(dataframe, punctuation, join_str):
    """
    Funzione che elabora il dataset per essere usato dal modello
    :param dataframe: dataframe d'input
    :type dataframe: pd.DataFrame
    :param punctuation: punti da eliminare
    :type punctuation: list
    :param join_str: stringa usata per effettuare l0unione dei token (riconvertendo il testo da lista a stringa)
    :type join_str: str
    :return: dataframe rielaborato
    :rtype: pd.DataFrame
    """
    dataframe = dataset_opening_preprocessing(dataframe)
    dataframe['text'] = text_processing(dataframe['text'], punctuation, join_str=join_str)
    dataframe['parent'] = text_processing(dataframe['parent'], punctuation, join_str=join_str)
    return dataframe


def max_sentence_len(text):
    """
    Funzione che restituisce la lunghezza massima delle sentenze
    il valore massimo viene calcolato eliminando le sentenze troppo lunghe (quelle outliers, trovate tramite l'IQR)
    :param text: testo d'input
    :type text: pd.Series
    :return: lunghezza massima
    :rtype: int
    """
    text_s_len = text.apply(len)
    q1, q3 = text_s_len.quantile(0.25), text_s_len.quantile(0.75)
    return round(2 * (q3 + 1.5 * (q3 - q1)))


df_train['parent'] = text_processing(df_train['parent'], del_punctuation)  # Processo il parent come il testo normale

# Calcolo il numero di token massimo nella distribuzione di testo e parent (parametro utilizzato dal modello)
text_len = max_sentence_len(df_train['text'])
parent_len = max_sentence_len(df_train['parent'])

if constants.ENABLE_OUT:
    print("lunghezza massima testo: ", text_len)
    print("lunghezza massima parent: ", parent_len)

# Preparo i dati di train per il modello
df_train['text'] = df_train['text'].apply(lambda words_list: " ".join(words_list))
df_train['parent'] = df_train['parent'].apply(lambda words_list: " ".join(words_list))
df_val = dataset_processing(df_val, del_punctuation, " ")

df_train.to_csv(os.path.join(constants.DATA_OUT_PATH, "train_processed.csv"))
df_val.to_csv(os.path.join(constants.DATA_OUT_PATH, "val_processed.csv"))


# ## Fase di modellazione
# In questa fase verrà definita e addestrata la rete neurale adottata
# """
#
#
# def custom_split(input_str):
#     return tensorflow.strings.split(input_str, sep=" <> ")
#
#
# def create_embedding_matrix(glove_path, vocab, embedding_dim, stemming=True):
#     """
#     Funzione che crea la embedding matrix a <embedding_dim> usando un GloVe Pretrained Embedding
#     :param glove_path: path al glove file
#     :type glove_path: str
#     :param vocab: dizionario contenente i termini usati
#     :type vocab: dict
#     :param embedding_dim: dimensione di embedding (deve essere compatibile con il file)
#     :type embedding_dim: int
#     :param stemming: indica se effettuare lo stemming delle parole
#     :type stemming: bool
#     :return: embedding matrix risultante
#     :rtype: np.matrix
#     """
#     stemmer = LancasterStemmer()
#     word_index = dict(zip(vocab, range(len(vocab))))
#     embedding_matrix = np.zeros((len(vocab), embedding_dim))
#
#     with open(glove_path, encoding="utf8") as glove_f:
#         for line in glove_f:
#             word, *vector = line.split()
#             if stemming:
#                 word = stemmer.stem(word)
#             if word in word_index.keys():
#                 embedding_matrix[word_index[word]] = np.array(vector, dtype='float32')
#
#     return embedding_matrix
#
#
# def create_vocabulary_mh(text, max_tokens):
#     """
#     Funzione che crea il vocabolario per un testo in modalità multi-hot
#
#     :param text: testo da processare
#     :type text: pd.Series
#     :param max_tokens: dimensione massima del vocabolario
#     :type max_tokens: int
#     :return: vocabolario
#     :rtype: np.array
#     """
#
#     text_vectorizer = layers.TextVectorization(
#         max_tokens=max_tokens,
#         standardize=None,
#         split=None,
#         output_mode='multi_hot'
#     )
#
#     text_vectorizer.adapt(text)
#     return text_vectorizer.get_vocabulary()
#
#
# def create_vocabulary_int(text, max_tokens, out_seq_len, glove_path, embedding_dim, split_f=None, stemming=True):
#     """
#     Funzione che crea il vocabolario per un testo in modalità int e genera la relativa embedding matrix usando un GloVe
#     :param text: testo da processare
#     :type text: pd.Series
#     :param max_tokens: dimensione massima del vocabolario
#     :type max_tokens: int
#     :param out_seq_len: dimensione massima di una sentenza
#     :type out_seq_len: int
#     :param split_f: funzione di split
#     :type split_f: function or None
#     :param glove_path: path al glove file
#     :type glove_path: str
#     :param embedding_dim: dimensione di embedding (deve essere compatibile con il file)
#     :type embedding_dim: int
#     :param stemming: indica se effettuare lo stemming delle parole
#     :type stemming: bool
#     :return: vocabolario, embedding
#     :rtype: (np.array, np.matrix)
#     """
#
#     text_vectorizer = layers.TextVectorization(
#         max_tokens=max_tokens,
#         standardize=None,
#         split=split_f,
#         output_sequence_length=out_seq_len,
#         output_mode='int'
#     )
#
#     text_vectorizer.adapt(text)
#
#     vocab = text_vectorizer.get_vocabulary()
#     embedding_matrix = create_embedding_matrix(glove_path, vocab, embedding_dim, stemming=stemming)
#
#     return vocab, embedding_matrix
#
#
# class SarcasmHyperModel(keras_tuner.HyperModel):
#     def __init__(self, df_train, name=None, tunable=True):
#         super().__init__(name, tunable)
#
#         max_tokens_choice = {"text": [1000, 10000, 20000], "parent": [1000, 10000, 20000],
#                              "author": [100, 1000, 2000], "subreddit": [100, 1000, 2000]}
#
#         text_vocab_m = {
#             max_tokens: create_vocabulary_int(df_train['text'], max_tokens, text_len, GLOVE_PATH, EMBEDDING_DIM,
#                                               split_f=custom_split) for max_tokens in max_tokens_choice['text']}
#         parent_vocab_m = {max_tokens: create_vocabulary_int(df_train['parent'], max_tokens, parent_len, GLOVE_PATH,
#                                                             EMBEDDING_DIM, split_f=custom_split) for
#                           max_tokens in max_tokens_choice['parent']}
#
#         author_vocab = {max_tokens: create_vocabulary_mh(df_train['author'], max_tokens) for
#                         max_tokens in max_tokens_choice['author']}
#         subreddit_vocab = {max_tokens: create_vocabulary_mh(df_train['subreddit'], max_tokens) for
#                            max_tokens in max_tokens_choice['subreddit']}
#
#         self.vocabs = {"text": text_vocab_m, "parent": parent_vocab_m,
#                        "author": author_vocab, "subreddit": subreddit_vocab}
#         self.lens = {"text": text_len, "parent": parent_len}
#
#     def build(self, hp):
#         tokens_text = hp.Choice("max_tokens_text", list(self.vocabs['text'].keys()))
#         tokens_parent = hp.Choice("max_tokens_parent", list(self.vocabs['parent'].keys()))
#         tokens_author = hp.Choice("max_tokens_author", list(self.vocabs['author'].keys()))
#         tokens_subreddit = hp.Choice("max_tokens_subreddit", list(self.vocabs['subreddit'].keys()))
#
#         text_units = hp.Int("text.processing_units", min_value=25, max_value=200, step=25)
#         parent_units = hp.Int("parent.processing_units", min_value=25, max_value=100, step=25)
#         author_units = hp.Int("author.processing_units", min_value=10, max_value=50, step=10)
#         subreddit_units = hp.Int("subreddit.processing_units", min_value=10, max_value=50, step=10)
#
#         text_input = layers.Input(shape=(1,), name='text', dtype=tensorflow.string)
#         parent_input = layers.Input(shape=(1,), name='parent', dtype=tensorflow.string)
#         author_input = layers.Input(shape=(1,), name='author', dtype=tensorflow.string)
#         subreddit_input = layers.Input(shape=(1,), name='subreddit', dtype=tensorflow.string)
#
#         text_layers = layers.TextVectorization(
#             max_tokens=tokens_text,
#             standardize=None,
#             split=custom_split,
#             output_mode='int',
#             output_sequence_length=self.lens['text'],
#             vocabulary=self.vocabs['text'][tokens_text][0],
#             name="text.int_encoding"
#         )(text_input)
#
#         parent_layers = layers.TextVectorization(
#             max_tokens=tokens_parent,
#             standardize=None,
#             split=custom_split,
#             output_mode='int',
#             output_sequence_length=self.lens['parent'],
#             vocabulary=self.vocabs['parent'][tokens_parent][0],
#             name="parent.int_encoding"
#         )(parent_input)
#
#         author_layers = layers.TextVectorization(
#             max_tokens=tokens_author,
#             standardize=None,
#             split=None,
#             output_mode='multi_hot',
#             vocabulary=self.vocabs['author'][tokens_author],
#             name="author.hot_encoding"
#         )(author_input)
#
#         subreddit_layers = layers.TextVectorization(
#             max_tokens=tokens_subreddit,
#             standardize=None,
#             split=None,
#             output_mode='multi_hot',
#             vocabulary=self.vocabs['subreddit'][tokens_subreddit],
#             name="subreddit.hot_encoding"
#         )(subreddit_input)
#
#         text_layers = layers.Embedding(output_dim=EMBEDDING_DIM,
#                                        input_length=self.lens['text'],
#                                        input_dim=len(self.vocabs['text'][tokens_text][0]),
#                                        mask_zero=True,
#                                        embeddings_initializer=initializers.Constant(
#                                            self.vocabs['text'][tokens_text][1]),
#                                        trainable=False,
#                                        name="text.glove_embedding"
#                                        )(text_layers)
#         parent_layers = layers.Embedding(output_dim=EMBEDDING_DIM,
#                                          input_length=self.lens['parent'],
#                                          input_dim=len(self.vocabs['parent'][tokens_parent][0]),
#                                          mask_zero=True,
#                                          embeddings_initializer=initializers.Constant(
#                                              self.vocabs['parent'][tokens_parent][1]),
#                                          trainable=False,
#                                          name="parent.glove_embedding"
#                                          )(parent_layers)
#
#         text_layers = layers.Bidirectional(layers.LSTM(text_units, return_sequences=True, name="lstm_tp1"),
#                                            name="text.processing_1")(text_layers)
#         text_layers = layers.Bidirectional(layers.LSTM(int(text_units / 5), return_sequences=False, name="lstm_tp2"),
#                                            name="text.processing_2")(text_layers)
#
#         parent_layers = layers.Bidirectional(layers.LSTM(parent_units, return_sequences=False,
#                                                          name="lstm_pp1"), name="parent.processing")(parent_layers)
#         author_layers = layers.Dense(author_units, name="author.processing__relu",
#                                      activation=activations.relu)(author_layers)
#         subreddit_layers = layers.Dense(subreddit_units, name="subreddit.processing__relu",
#                                         activation=activations.relu)(subreddit_layers)
#
#         contex_layers = layers.Concatenate(name="contex_concatenation")(
#             [parent_layers, author_layers, subreddit_layers])
#         context_units = round((parent_units + author_units + subreddit_units) / 10)
#         contex_layers = layers.Dense(context_units,
#                                      name="contex.processing__relu", activation=activations.relu)(contex_layers)
#
#         global_layers = layers.Concatenate(name="global_concatenation")([text_layers, contex_layers])
#         global_units = round((context_units + text_units) / 4)
#         global_layers = layers.Dense(global_units, name="global.processing__relu",
#                                      activation=activations.relu)(global_layers)
#
#         output = layers.Dense(1, activation=activations.sigmoid, name="output")(global_layers)
#         model = models.Model(inputs=[text_input, parent_input, author_input, subreddit_input], outputs=output)
#         model.compile(optimizer="adam", loss=losses.BinaryCrossentropy(), metrics=[metrics.BinaryAccuracy()])
#         return model
#
#     def fit(self, hp, model, *args, **kwargs):
#         return model.fit(*args, **kwargs)
#
#
# """Creazione e addestramento del modello"""
#
# EPOCHS = 20
#
# tuner_bayesian = keras_tuner.BayesianOptimization(
#     hypermodel=SarcasmHyperModel(df_train),
#     objective="val_binary_accuracy",
#     max_trials=10,
#     executions_per_trial=1,
#     overwrite=True,
#     directory="ris/models",
#     project_name="bayesian_tuner"
# )
#
# tuner_bayesian.search(x=[df_train['text'], df_train['parent'], df_train['author'], df_train['subreddit']],
#                       y=df_train[TARGET],
#                       validation_data=([df_val['text'], df_val['parent'], df_val['author'], df_val['subreddit']],
#                                        df_val[TARGET]),
#                       batch_size=128, epochs=EPOCHS, verbose=1,
#                       callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=int(EPOCHS / 10),
#                                                          restore_best_weights=True),
#                                  callbacks.TensorBoard("ris/logs")])
# tuner_bayesian.results_summary()
#
# """Plotting dell'history del modello"""
#
# history = pd.DataFrame(data=history)
# history.index.name = 'epochs'
#
# history.plot(title="History di training")
# plt.show()
#
# """Try"""
#
# compare_df = df_val[TARGET].to_frame(name="real")
# compare_df['predicted'] = np.round(model.predict([df_val['text'], df_val['parent'],
#                                                   df_val['author'], df_val['subreddit']])).astype(int)
#
# print(compare_df)
# from sklearn.metrics import classification_report
#
# print(classification_report(compare_df['real'], compare_df['predicted'], target_names=['sarcastic', 'not sarcastic']))
#
# from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay
#
# conf_matrix = confusion_matrix(compare_df['real'], compare_df['predicted'])
# print(conf_matrix)
#
# PrecisionRecallDisplay.from_predictions(compare_df['real'], compare_df['predicted']).plot()
#
# import seaborn as sns
#
# fig, ax = plt.subplots()
# sns.heatmap(conf_matrix, annot=True, ax=ax)