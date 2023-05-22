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
import os
import constants



"""Dowload dei file per nltk"""

# nltk.download('stopwords')


"""Definizione di alcune costanti"""



"""### Fase di import del dataset e prima analisi
In questa fase verrà importato il dataset (suddividendolo in train e validation set) e si analizzerà:
- Il numero di righe, la presenza di duplicate e di nulle
- La distribuzione del target
- La presenza di elementi del contesto ripetuti

"""

df_full = pd.read_csv(os.path.join(constants.DATA_PATH, "data_full.tsv"),
                      sep="\t", names=[constants.TARGET, "text", "author", "subreddit", "date", "parent"]).sample(frac=0.05)

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
df_train.to_csv(os.path.join(constants.DATA_PATH, "train.csv"))

if constants.ENABLE_OUT:
    print("tipi di variabile dopo la conversione:\n", df_train.dtypes, "\n")
    # Analisi del target
    print("Stampa di 3 righe sarcastiche:\n", df_train.loc[df_train[constants.TARGET] == 1].head(3)[[constants.TARGET, 'text']], "\n")
    print("Stampa di 3 righe non sarcastiche:\n", df_train.loc[df_train[constants.TARGET] == 0].head(3)[[constants.TARGET, 'text']], "\n\n")
    print("Distribuzione del target:")
    print(df_train[constants.TARGET].value_counts(normalize=True))

# """### Analisi di elementi ripetuti nel contesto, utile per verificare se essi possano essere fonte d'informazione
# Si definiscono, e poi applicano, due funzioni a tal proposito:
# """
#
#
# def compute_proportions_series(s1, s2):
#     """
#     Funzione, di appoggio, che prende due serie intere con indici simili (alcuni elementi in comune, ma non tutte)
#     ed effettua la proporzione della prima sull'altra
#     :param s1: prima serie d'input
#     :type s1: pd.Series
#     :param s2: seconda serie d'input
#     :type s2: pd.Series
#     :return: Dataframe risultate contente le proporzioni e il numero totale di elementi
#     :rtype: pd.DataFrame
#     """
#
#     s1_r = s1.reindex(s1.index.join(s2.index, how="outer"), fill_value=0)
#     s2_r = s2.reindex(s1.index.join(s2.index, how="outer"), fill_value=0)
#
#     df_out = pd.DataFrame(columns=['proportion', 'tot'], index=s1_r.index)
#     df_out['tot'] = s1_r + s2_r
#     df_out['proportion'] = s1_r / df_out['tot']
#
#     return df_out
#
#
# def compute_sarcastic_unique_stats(dataframe, thresholds):
#     """
#     Funzione che calcola per ogni elemento del contesto il value_counts sarcastico e non.
#     Calcola inoltre il numero di elementi unici che superano delle soglie di sarcasmo,
#     e restituisce anche le loro proporzioni
#
#     :param dataframe: dataframe di elaborazione
#     :type dataframe: pd.DataFrame
#     :param thresholds: soglie in cui calcolare il numero di elementi unici sarcastici
#     :type thresholds: list
#     :return: sarcastic_vc, no_sarcastic_vc, sarcastic_proportions, sarcastic_unique_stats
#     :rtype: (pd.Series, pd.Series, pd.Series, pd.DataFrame)
#     """
#
#     dataframe_s = dataframe.loc[dataframe[TARGET] == 1]
#     dataframe_ns = dataframe.loc[dataframe[TARGET] == 0]
#
#     sarcastic_vc = pd.Series(data=[dataframe_s[col].value_counts() for col in CONTEXT_COLS], index=CONTEXT_COLS)
#     no_sarcastic_vc = pd.Series(data=[dataframe_ns[col].value_counts() for col in CONTEXT_COLS], index=CONTEXT_COLS)
#     sarcastic_proportions = pd.Series(index=CONTEXT_COLS, dtype="object")
#
#     unique_stats = pd.DataFrame(columns=CONTEXT_COLS, index=thresholds)
#
#     tot = dataframe[CONTEXT_COLS].nunique()
#
#     for col in CONTEXT_COLS:
#         proportions = compute_proportions_series(sarcastic_vc[col], no_sarcastic_vc[col])
#         sarcastic_proportions[col] = proportions.sort_values(by='proportion', ascending=False)
#
#         unique_stats[col] = unique_stats[col].to_frame().apply(
#             lambda row: (proportions['proportion'] >= row.name / 100).sum(), axis="columns")
#
#     unique_stats.loc['tot unique'] = tot
#
#     return sarcastic_vc, no_sarcastic_vc, sarcastic_proportions, unique_stats
#
#
# df_s_vc, df_ns_vc, s_prop, df_unique_stats = compute_sarcastic_unique_stats(df_train, [100, 75, 50, 25])
#
# if ENABLE_OUT:
#     print("\n\n\nAnalisi del numero di subreddit, autori e parent unici:\n", df_unique_stats, "\n")
#     print("In percentuale:\n", df_unique_stats * 100 / len(df_train), "\n")
#     print("In percentuale rispetto alla categoria:\n", df_unique_stats.iloc[:-1] * 100 / df_unique_stats.iloc[-1], "\n")
#
# """Si individua la presenza di elementi duplicati nel contesto, principalmente per la feature 'subreddit'.
# Si procede dunque con un'approfondimento su essa
# """
#
# subreddit_s_prop = s_prop['subreddit'].loc[s_prop['subreddit']['proportion'] >= 0.75]
# # si considerano i subreddit con più di un post e si eliminano gli outlier per maggiore chiarezza
# subreddit_s_prop = subreddit_s_prop.loc[subreddit_s_prop['tot'] > 1]
# subreddit_s_prop_outliers = subreddit_s_prop.loc[abs(zscore(subreddit_s_prop['tot'])) >= 3]
# subreddit_s_prop = subreddit_s_prop.loc[abs(zscore(subreddit_s_prop['tot'])) < 3]
#
# if ENABLE_OUT:
#     print("Proporzioni e num post di subreddit con numero di post > di 1:\n", subreddit_s_prop,
#           "\ngli outliers sono:\n", subreddit_s_prop_outliers, "\n")
#     plt.scatter(subreddit_s_prop['proportion'], subreddit_s_prop['tot'])
#     plt.title("Scatter che mostra la presenza di subreddit a maggioranza sarcastici")
#     plt.xlabel("Rateo Sarcastici/Totali")
#     plt.ylabel("Numero di post")
#     plt.show()
#
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

Si definiscono due funzioni generiche a tal proposito:
"""


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


def print_plot_most_common_token(series, num_print=10, num_plot=20,
                                 text_print="Frequenza nel testo dei token più comuni:",
                                 title_plot="Frequenza nel testo dei token più comuni"):
    """
    Funzione che stampa e plotta i token più comuni in una serie contente i token
    :param series: serie d'input
    :type series: pd.Series
    :param num_print: numero di token da stampare
    :type num_print: int
    :param num_plot: numero di token da usare nei plot
    :type num_plot: int
    :param text_print: testo da stampare a schermo
    :type text_print: str
    :param title_plot: titolo del plot
    :type title_plot: str
    :return: serie dei token più comune
    :rtype pd.Series
    """
    common_tokens = values_count_from_list(series, normalize=True)
    print("\n", text_print, "\n", common_tokens.head(num_print), "\n\n")
    fig, ax = plt.subplots()
    (common_tokens.head(num_plot) * 100).plot(kind='bar', title=title_plot, xlabel="Token", ylabel="Frequenza %")
    fig, ax = plt.subplots()
    plt.imshow(WordCloud(width=1600, height=800,
                         background_color="black").generate_from_frequencies(dict(common_tokens.to_dict())),
               interpolation='antialiased')


"""Si procede con la tokenizzazione del testo"""

tweet_tokenizer = TweetTokenizer()

df_train['text_tokenized'] = df_train['text'].apply(lambda x: tweet_tokenizer.tokenize(x))

if constants.ENABLE_OUT:
    print("stampa di tre frasi con i relativi token:\n", df_train[['text', 'text_tokenized']].head(3), "\n\n")
    print_plot_most_common_token(df_train['text_tokenized'])
    plt.show()

"""Prima di eliminare la punteggiatura, si esamina la frequenza dei simboli in frasi sarcastiche e non, in quanto essi possono essere fonte d'informazione."""

all_punctuation = list(string.punctuation)
all_punctuation.append("...")
punctuation_freq = pd.DataFrame(columns=["sarcastic", "non_sarcastic"], index=all_punctuation, dtype="float64")
punctuation_freq['sarcastic'] = punctuation_freq.apply(
    lambda x: df_train.loc[df_train[constants.TARGET] == 1, 'text'].str.contains(re.escape(x.name)).sum(), axis="columns")
punctuation_freq['non_sarcastic'] = punctuation_freq.apply(
    lambda x: df_train.loc[df_train[constants.TARGET] == 0, 'text'].str.contains(re.escape(x.name)).sum(), axis="columns")
punctuation_freq['sarcastic'] = punctuation_freq['sarcastic'] * 100 / (df_train[constants.TARGET] == 1).sum()
punctuation_freq['non_sarcastic'] = punctuation_freq['non_sarcastic'] * 100 / (df_train[constants.TARGET] == 0).sum()
punctuation_freq['rateo'] = round(punctuation_freq['sarcastic'] / punctuation_freq['non_sarcastic'], 4).fillna(0)
punctuation_freq = punctuation_freq.sort_values(by='rateo', ascending=False)

punctuation_freq.to_csv(os.path.join(constants.DATA_PATH, "punctuation_freq.csv"))

if constants.ENABLE_OUT:
    print("Frequenza della punteggiatura in frasi sarcastiche:\n", punctuation_freq, "\n\n")
    plt.subplots()
    punctuation_freq['rateo'].plot(kind='bar', title="Frequenza della punteggiatura in frasi sarcastiche")
    plt.show()


"""Vista la distribuzione del rateo dei simboli, si decide di mantenere quelli che sono degli outliers alla distribuzione (perchè possono discriminare meglio una frase sarcastica o non)"""

# Rimozione
outlier_punctuation = punctuation_freq.loc[abs(zscore(punctuation_freq['rateo'])) >= 3].index.values
del_punctuation = [point for point in list(all_punctuation) if point not in outlier_punctuation]
df_train['text_tokenized'] = df_train['text_tokenized'].apply(
    lambda word_list: [word for word in word_list if word not in del_punctuation])

if constants.ENABLE_OUT:
    print("I punti mantenuti sono:\t", outlier_punctuation)
    print_plot_most_common_token(df_train['text_tokenized'], text_print="Dopo la rimozione della punteggiatura:",
                                 title_plot="Dopo la rimozione della punteggiatura")
    plt.show()

"""Si procede con l'eliminazione delle stopwords"""

df_train['text_nsw'] = df_train['text_tokenized'].apply(
    lambda word_list: [word for word in word_list if word not in stopwords.words('english')])

if constants.ENABLE_OUT:
    print_plot_most_common_token(df_train['text_nsw'], text_print="Dopo la rimozione delle stopword:",
                                 title_plot="Dopo la rimozione delle stopword")
    plt.show()

"""Si termina la fase con lo stemming"""

# Stemming
stemmer = LancasterStemmer()
df_train['text_nsw_st'] = df_train['text_nsw'].apply(
    lambda word_list: [stemmer.stem(word) for word in word_list])

# Stemming con stopwords
df_train['text_st'] = df_train['text_tokenized'].apply(lambda word_list: [stemmer.stem(word) for word in word_list])

train_text = df_train[['text_tokenized', 'text_nsw', 'text_nsw_st', 'text_st']].rename({
    'text_tokenized': 'tokenized', 'text_nsw': 'nsw', 'text_nsw_st': 'nsw_st', 'text_st': 'st'}, axis='columns')

train_text.to_csv(os.path.join(constants.DATA_PATH, "train_text.csv"))

if constants.ENABLE_OUT:
    print_plot_most_common_token(df_train['text_nsw_st'], text_print="Dopo la rimozione delle stopword e stemming:",
                                 title_plot="Dopo la rimozione delle stopword e stemming")
    print_plot_most_common_token(df_train['text_st'], text_print="Dopo lo stemming:", title_plot="Dopo lo stemming")
    plt.show()

"""## Confronto dei tipi di testo
In questa fase verranno confrontati i tre tipi di testi prodotti (nsw, nsw_st, st), in modo da individuare quale di essi porta più informazioni.

"""


def compute_helpful_words(text, target, vocabulary_size=1000, z_score=3):
    """
    Funzione che calcola le parole più utili per un modello, calcolando le proporzioni in cui compaiono sarcastiche e
    non, per poi mantenere quelle che superano lo z score
    :param text: serie contente il testo da elaborare (sotto forma di lista di token)
    :type text: pd.Series
    :param target: serie del target che (avendo lo stesso indice di text) fornisce la label
    :type target: pd.Series
    :param vocabulary_size: numero di feature-parole da estrarre
    :type vocabulary_size: int or None
    :param z_score: precisione di esclusione
    :type z_score: float
    :return: dataframe che ha come indice le parole, e come valori le loro proporzioni
    :rtype: pd.DataFrame
    """
    text_vectorizer = CountVectorizer(max_features=vocabulary_size)
    text = text.apply(lambda words_list: " ".join(words_list))

    text_vectorized = text_vectorizer.fit_transform(text)
    target = target.to_frame(constants.TARGET)
    target['sparse_index'] = np.arange(len(target))

    index_s = target.loc[target[constants.TARGET] == 1, 'sparse_index']
    index_ns = target.loc[target[constants.TARGET] == 0, 'sparse_index']
    text_vectorized_s, text_vectorized_ns = text_vectorized[index_s.values] > 0, text_vectorized[index_ns.values] > 0

    text_s_prop = pd.DataFrame(index=text_vectorizer.get_feature_names_out(), columns=['sarcastic', 'not_sarcastic'])
    text_s_prop['sarcastic'] = np.array(text_vectorized_s.sum(axis=0))[0] / len(index_s)
    text_s_prop['not_sarcastic'] = np.array(text_vectorized_ns.sum(axis=0))[0] / len(index_ns)
    text_s_prop['rate'] = text_s_prop['sarcastic'] / (text_s_prop['sarcastic'] + text_s_prop['not_sarcastic'])
    text_s_prop['tot_occ'] = (text_s_prop['sarcastic'] * len(index_s) +
                              text_s_prop['not_sarcastic'] * len(index_ns))

    # elimino le parole che occorrono poco
    text_s_prop = text_s_prop.loc[text_s_prop['tot_occ'] > text_s_prop['tot_occ'].quantile(0.3)]

    return text_s_prop.loc[abs(zscore(text_s_prop['rate'])) >= z_score].sort_values(by='rate', ascending=False)


precision = 2
vocab_size = None

nsw_hw = compute_helpful_words(df_train['text_nsw'], df_train[constants.TARGET], vocabulary_size=vocab_size, z_score=precision)
nsw_st_hw = compute_helpful_words(df_train['text_nsw_st'], df_train[constants.TARGET], vocabulary_size=vocab_size,
                                  z_score=precision)
st_hw = compute_helpful_words(df_train['text_st'], df_train[constants.TARGET], vocabulary_size=vocab_size, z_score=precision)

nsw_hw.to_csv(os.path.join(constants.DATA_PATH, "train_text_hp", "nsw_hw.csv"))
nsw_st_hw.to_csv(os.path.join(constants.DATA_PATH, "train_text_hp", "nsw_st_hw.csv"))
st_hw.to_csv(os.path.join(constants.DATA_PATH, "train_text_hp", "st_hw.csv"))


if constants.ENABLE_OUT:
    print("nsw:\n", nsw_hw, "\n\nnsw_st:\n", nsw_st_hw, "\n\nst:\n", st_hw)
    print("\n\nIl migliore è:\t",
          pd.Series(index=['nsw', 'nsw_st', 'st'], data=[len(text) for text in [nsw_hw, nsw_st_hw, st_hw]]
                    ).sort_values(ascending=False))

"""Vista l'uguaglianza dei tre, si decide di usare il testo senza stopwords e stemming (in quanto di dimensione ridotta)

"""

df_train = df_train.drop(columns=['text', 'text_tokenized', 'text_nsw', 'text_st']
                         ).rename(columns={'text_nsw_st': 'text'})

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

    stem = LancasterStemmer()
    nsw_st = nsw.apply(lambda word_list: [stem.stem(word) for word in word_list])

    if join_str is not None:
        nsw_st = nsw_st.apply(lambda words_list: join_str.join(words_list))

    return nsw_st


def dataset_processing(dataframe, punctuation):
    """
    Funzione che elabora il dataset per essere usato dal modello
    :param dataframe: dataframe d'input
    :type dataframe: pd.DataFrame
    :param punctuation: punti da eliminare
    :type punctuation: list
    :return: dataframe rielaborato
    :rtype: pd.DataFrame
    """
    dataframe = dataset_opening_preprocessing(dataframe)
    dataframe['text'] = text_processing(dataframe['text'], punctuation, join_str=" <> ")
    dataframe['parent'] = text_processing(dataframe['parent'], punctuation, join_str=" <> ")
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
df_train['text'] = df_train['text'].apply(lambda words_list: " <> ".join(words_list))
df_train['parent'] = df_train['parent'].apply(lambda words_list: " <> ".join(words_list))
df_val = dataset_processing(df_val, del_punctuation)

df_train.to_csv(os.path.join(constants.DATA_PATH, "train_processed.csv"))
df_val.to_csv(os.path.join(constants.DATA_PATH, "val_processed.csv"))


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