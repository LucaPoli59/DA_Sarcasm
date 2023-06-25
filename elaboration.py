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
from nltk import LancasterStemmer, TweetTokenizer
from nltk.corpus import stopwords
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from plotly.express.colors import sample_colorscale
import os
import constants
import json

import timeit

"""Dowload dei file per nltk"""

# nltk.download('stopwords')


"""Definizione di alcune costanti"""



"""### Fase di import del dataset e prima analisi
In questa fase verrà importato il dataset (suddividendolo in train e validation set) e si analizzerà:
- Il numero di righe, la presenza di duplicate e di nulle
- La distribuzione del target
- La presenza di elementi del contesto ripetuti

"""

start_timer = timeit.default_timer()

df_full = pd.read_csv(os.path.join(constants.DATA_IN_PATH, "data_full.tsv"),
                      sep="\t", names=[constants.TARGET, "text", "author", "subreddit", "date", "parent"])

df_full.to_csv(os.path.join(constants.DATA_OUT_PATH, "data_full_sample.csv"))

df_train, df_val = train_test_split(df_full, test_size=0.05)

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

    if join_str is not None:
        nsw = nsw.apply(lambda words_list: join_str.join(words_list))

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



# Preparo i dati di test per il modello
df_test = pd.read_csv(os.path.join(constants.DATA_IN_PATH, "test.tsv"),
                      sep="\t", names=[constants.TARGET, "text", "author", "subreddit", "date", "parent"])

df_test = dataset_processing(dataset_opening_preprocessing(df_test), del_punctuation, " ")
df_test.to_csv(os.path.join(constants.DATA_OUT_PATH, "test_processed.csv"))

end_timer = timeit.default_timer()
print("Tempo di esecuzione: ", end_timer - start_timer, " secondi")
