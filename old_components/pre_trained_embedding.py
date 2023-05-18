# import silence_tensorflow.auto
import pandas as pd
from random import randint
import numpy as np
import tensorflow
from nltk import LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import metrics, losses, layers, activations, models, callbacks, utils
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


# N.B. Qua uso il tokenizer sulle frasi, ma è meglio usarlo sui token risultati dall'altra fase
df = pd.read_json("dataset/train-processed.json", encoding='utf-8')


random_state = randint(0, 1000)
test_size = 0.1

target_train, target_val = train_test_split(df[TARGET], random_state=random_state, test_size=test_size)
nsw_st_train, nsw_st_val = train_test_split(df['text_nsw_st'], test_size=test_size, random_state=random_state)


# Calcolo di max_tokens

text_train = nsw_st_train.apply(lambda words_list: " <> ".join(words_list))
text_val = nsw_st_val.apply(lambda words_list: " <> ".join(words_list))

vectorize_layer = layers.TextVectorization(
    max_tokens=None,
    standardize=None,
    split=custom_split,
    output_mode='int',
    output_sequence_length=20,
    name="vectorizer"
)

vectorize_layer.adapt(text_train)
vocabulary = vectorize_layer.get_vocabulary()

timer_start = timeit.default_timer()

embedding_matrix_twitter = create_embedding_matrix(ALT_DIR + "glove/glove.twitter.27B.100d.txt", vocabulary, 100)
embedding_matrix_std = create_embedding_matrix(ALT_DIR + "glove/glove.42B.300d.txt", vocabulary, 300)
embedding_matrix_std_6 = create_embedding_matrix(ALT_DIR + "glove/glove.6B.300d.txt", vocabulary, 300)


print(embedding_matrix_twitter)
print(embedding_matrix_std)
print(embedding_matrix_std_6)

twitter_score = np.count_nonzero(np.count_nonzero(embedding_matrix_twitter, axis=1)) / len(vocabulary)
std_score = np.count_nonzero(np.count_nonzero(embedding_matrix_std, axis=1)) / len(vocabulary)
std_6_score = np.count_nonzero(np.count_nonzero(embedding_matrix_std_6, axis=1)) / len(vocabulary)

train_vc = values_count_from_list(nsw_st_train)
num_one_time_word = len(train_vc.loc[train_vc == 1])

# numero di righe di parole nel dizionario di cui abbiamo trovato il vettore corrispondente
print("L'Embedding matrix con GloVe-Twitter è completa al ", round(twitter_score, 2) * 100, "%")
print("L'Embedding matrix con GloVe-Standard è completa al ", round(std_score, 2) * 100, "%")
print("L'Embedding matrix con GloVe-Standard_6 è completa al ", round(std_6_score, 2) * 100, "%")
print("Si nota però che il ", round(num_one_time_word * 100 / len(vocabulary), 2),
      "% di parole si ripetono una sola volta")

# with open('dataset/glove_twitter.json', 'w', encoding='utf-8') as json_df:
#     pd.DataFrame(embedding_matrix_twitter).to_json(json_df, force_ascii=False)
#
# with open('dataset/glove_std.json', 'w', encoding='utf-8') as json_df:
#     pd.DataFrame(embedding_matrix_std).to_json(json_df, force_ascii=False)



