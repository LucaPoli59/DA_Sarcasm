import string
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from nltk.tokenize import sent_tokenize, TweetTokenizer, toke
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from scipy.stats import zscore

# '''''''''''''''''' Settings '''''''''''''''''''''


TARGET = "sarcastic"
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.grid'] = True
pd.set_option('display.max_colwidth', 100)

# '''''''''''''' Custom Function '''''''''''


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
    Funzione che stampa e plotta i token più comuni
    :param text_print:
    :type text_print:
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


# '''''''''''''''''' Text Processing '''''''''''''''''''''


df = pd.read_csv("dataset/train-balanced.tsv", sep="\t", names=["sarcastic", "text", "author", "subreddit",
                                                                "date", "parent"])
df.index.name = "index"
df = df.dropna()
df['sarcastic'] = df['sarcastic'].astype("category")
df[['text', 'parent', 'subreddit', 'author']] = df[["text", "parent", "subreddit", "author"]].astype("str")
df['date'] = pd.to_datetime(df['date'], format="%Y-%m")
df['text'] = df['text'].str.lower()
df['parent'] = df['parent'].str.lower()


# # Calcolo delle sotto sentence con il relativo tokenizer
# df['sentences'] = df['text'].apply(lambda x: sent_tokenize(x))
# df['n_sentences'] = df['sentences'].apply(len)
#
# print("stampa di tre frasi con multiple sotto frasi:\n",
#       df.loc[df['n_sentences'] > 1, ['text', 'sentences']].head(3), "\n\n")
#
# print("Distribuzione del numero di sentence:")
# print(round(df[['n_sentences', TARGET]].value_counts(normalize=True), 4))
# #pd.crosstab(df['n_sentences'], df[TARGET]).plot.bar(stacked=True)
# # possiamo notare che le occorrenze con numero di sentenze > 1 è irrilevante (è < dell' 1% del dataset)

# usiamo il tweet_tokenizer
tweet_tokenizer = TweetTokenizer()

df['text_tokenized'] = df['text'].apply(lambda x: tweet_tokenizer.tokenize(x))
print("stampa di tre frasi con i relativi token:\n", df[['text', 'text_tokenized']].head(3), "\n\n")
print_plot_most_common_token(df['text_tokenized'])

# Notiamo una forte presenza di stopwords e punteggiatura

# analizziamo la loro importanza vedendo se sono più frequenti in frasi sarcastiche o non
punctuation = list(string.punctuation)
punctuation.append("...")
punctuation_freq = pd.DataFrame(columns=["sarcastic", "non_sarcastic"], index=punctuation, dtype="float64")
punctuation_freq['sarcastic'] = punctuation_freq.apply(
    lambda x: df.loc[df[TARGET] == 1, 'text'].str.contains(re.escape(x.name)).sum(), axis="columns")
punctuation_freq['non_sarcastic'] = punctuation_freq.apply(
    lambda x: df.loc[df[TARGET] == 0, 'text'].str.contains(re.escape(x.name)).sum(), axis="columns")
punctuation_freq['sarcastic'] = punctuation_freq['sarcastic'] * 100 / (df[TARGET] == 1).sum()
punctuation_freq['non_sarcastic'] = punctuation_freq['non_sarcastic'] * 100 / (df[TARGET] == 0).sum()
punctuation_freq['rateo'] = round(punctuation_freq['sarcastic'] / punctuation_freq['non_sarcastic'], 4).fillna(0)
punctuation_freq = punctuation_freq.sort_values(by='rateo', ascending=False)

print("Frequenza della punteggiatura:\n", punctuation_freq, "\n\n")
#punctuation_freq['rateo'].plot(kind='bar')

outlier_punctuation = punctuation_freq.loc[abs(zscore(punctuation_freq['rateo'])) >= 3].index.values
print("I punti mantenuti sono:\t", outlier_punctuation, "\ncon i rispettivi bound:")

del_punctuation = [point for point in list(punctuation) if point not in outlier_punctuation]

# eliminazione della punteggiatura
df['text_tokenized'] = df['text_tokenized'].apply(
    lambda word_list: [word for word in word_list if word not in del_punctuation])

print_plot_most_common_token(df['text_tokenized'], text_print="Dopo la rimozione della punteggiatura:",
                             title_plot="Dopo la rimozione della punteggiatura")

# eliminazione delle stopwords
df['text_tokenized_nsw'] = df['text_tokenized'].apply(
    lambda word_list: [word for word in word_list if word not in stopwords.words('english')])
print_plot_most_common_token(df['text_tokenized_nsw'], text_print="Dopo la rimozione delle stopword:",
                             title_plot="Dopo la rimozione delle stopword")

stemmer = LancasterStemmer()
df['text_tokenized_nsw_st'] = df['text_tokenized_nsw'].apply(
    lambda word_list: [stemmer.stem(word) for word in word_list])
print_plot_most_common_token(df['text_tokenized_nsw_st'], text_print="Dopo la rimozione delle stopword e stemming:",
                             title_plot="Dopo la rimozione delle stopword e stemming")

df['text_tokenized_st'] = df['text_tokenized'].apply(lambda word_list: [stemmer.stem(word) for word in word_list])
print_plot_most_common_token(df['text_tokenized_st'], text_print="Dopo lo stemming:", title_plot="Dopo lo stemming")

plt.show()

# for col in ['text_tokenized_nsw', 'text_tokenized_nsw_st', 'text_tokenized_st']:
#     df[col] = df[col].apply(lambda words_list: " ".join(words_list))

df = df.rename(columns={'text_tokenized_nsw': 'text_nsw', 'text_tokenized_nsw_st': 'text_nsw_st',
                        'text_tokenized_st': 'text_st'})

# .drop(columns=["n_sentences", "sentences"])

with open('dataset/train-processed.json', 'w', encoding='utf-8') as json_df:
    df.to_json(json_df, force_ascii=False)

