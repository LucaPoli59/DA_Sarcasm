import json
import os
import re
import string

import pandas as pd
from nltk import LancasterStemmer, TweetTokenizer
from nltk.corpus import stopwords
from plotly.express.colors import sample_colorscale
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

import constants

"""Caricamento del dataset e preprocessing iniziale"""


df_full = pd.read_csv(os.path.join(constants.DATA_IN_PATH, "data_full.tsv"),
                      sep="\t", names=[constants.TARGET, "text", "author", "subreddit", "date", "parent"])

df_full.to_csv(os.path.join(constants.DATA_OUT_PATH, "data_full.csv"))

df_train, df_val = train_test_split(df_full, test_size=0.05)


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

"""Calcolo del rateo informativo per gli elementi del contesto"""


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


df_train_len = df_train[['sarcastic', 'text', 'parent']].copy()
df_train_len[['text', 'parent']] = df_train_len[['text', 'parent']].applymap(lambda x: len(x.split()))


for feature in ['text', 'parent']:
    sarc_prop = sarcastic_proportion_count(df_train_len.loc[abs(zscore(df_train_len[feature])) < 3,
                                           [constants.TARGET, feature]], target_info_rate)
    sarc_prop.to_csv(os.path.join(constants.DATA_SP_PATH, "len_" + feature + ".csv"))


# nota: si eliminano gli outliers per la lunghezza del testo e del parent
sarc_prop = sarcastic_proportion_count(df_train_len.loc[(abs(zscore(df_train_len[['text', 'parent']])) < 3).all(axis=1),
                                                        [constants.TARGET, 'text', 'parent']], target_info_rate)
sarc_prop.to_csv(os.path.join(constants.DATA_SP_PATH, "len_text_parent.csv"))


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


"""Fase di analisi del testo"""

tweet_tokenizer = TweetTokenizer()
df_train['text_tokenized'] = df_train['text'].apply(lambda x: tweet_tokenizer.tokenize(x))


# analisi della punteggiatura
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


# Rimozione della punteggiatura
outlier_punctuation = ['!']
del_punctuation = [point for point in list(all_punctuation) if point not in outlier_punctuation]
df_train['text_tokenized'] = df_train['text_tokenized'].apply(
    lambda word_list: [word for word in word_list if word not in del_punctuation])


# Eliminazione delle stopwords

df_train['text_nsw'] = df_train['text_tokenized'].apply(
    lambda word_list: [word for word in word_list if word not in stopwords.words('english')])


# Stemming
stemmer = LancasterStemmer()
df_train['text_nsw_st'] = df_train['text_nsw'].apply(
    lambda word_list: [stemmer.stem(word) for word in word_list])

# Stemming con stopwords
df_train['text_st'] = df_train['text_tokenized'].apply(lambda word_list: [stemmer.stem(word) for word in word_list])


# Confronto dei tipi di testo
for text_type in ['text_nsw', 'text_nsw_st', 'text_st', 'text_tokenized']:
    sarc_prop = sarcastic_proportion_count(df_train[[constants.TARGET, text_type]].explode(column=text_type),
                                           target_info_rate)
    sarc_prop.to_csv(os.path.join(constants.DATA_SP_PATH, text_type + ".csv"))
    word_cloud_generator(sarc_prop, text_type)


train_text = df_train[['text_tokenized', 'text_nsw', 'text_nsw_st', 'text_st']].rename({
    'text_tokenized': 'tokenized', 'text_nsw': 'nsw', 'text_nsw_st': 'nsw_st', 'text_st': 'st'}, axis='columns')

train_text = train_text.applymap(lambda x: " ".join(x))
train_text.to_csv(os.path.join(constants.DATA_OUT_PATH, "train_text.csv"))

df_train = df_train.drop(columns=['text', 'text_tokenized', 'text_nsw_st', 'text_st']
                         ).rename(columns={'text_nsw': 'text'})

"""## Fase di preprocessing finale
In questa fase avviene il preprocessing finale prima del modello:
- Si elabora il testo parent come il testo del commento
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


# Processo il parent come il testo normale
df_train['parent'] = text_processing(df_train['parent'], del_punctuation)


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

pd.Series(del_punctuation).to_csv(os.path.join(constants.MODEL_DIR, "del_punctuation.csv"))
