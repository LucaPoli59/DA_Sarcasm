import constants
import os
import pandas as pd
from tensorflow.python.keras import losses, metrics, activations
import keras_nlp
import keras
import tensorflow as tf

df_train = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train_processed.csv"), index_col='index'
                       ).sample(frac=0.1)
df_train['sarcastic'] = df_train['sarcastic'].astype(int)
df_train[df_train.columns[1:]] = df_train[df_train.columns[1:]].astype(str)

df_train['text_len'] = df_train['text'].apply(lambda x: len(x.split()))
df_train['text_len'] = df_train['text_len'] / df_train['text_len'].max()
df_train['parent_len'] = df_train['parent'].apply(lambda x: len(x.split()))
df_train['parent_len'] = df_train['parent_len'] / df_train['parent_len'].max()

df_val = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "val_processed.csv"), index_col='index'
                     ).sample(frac=0.1)
df_val['sarcastic'] = df_val['sarcastic'].astype(int)
df_val[df_val.columns[1:]] = df_val[df_val.columns[1:]].astype(str)

df_val['text_len'] = df_val['text'].apply(lambda x: len(x.split()))
df_val['text_len'] = df_val['text_len'] / df_val['text_len'].max()
df_val['parent_len'] = df_val['parent'].apply(lambda x: len(x.split()))
df_val['parent_len'] = df_val['parent_len'] / df_val['parent_len'].max()

author_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    standardize=None,
    split=None,
    output_mode='multi_hot',
    name="author.hot_encoding"
)
author_vectorizer.adapt(df_train['author'])

subreddit_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=1000,
    standardize=None,
    split=None,
    output_mode='multi_hot',
    name="subreddit.hot_encoding"
)
subreddit_vectorizer.adapt(df_train['subreddit'])

text_preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
parent_preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
text_backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
parent_backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
text_backbone.trainable, parent_backbone.trainable = False, False

text_input = tf.keras.Input(shape=(), dtype=tf.string, name='text')
parent_input = tf.keras.Input(shape=(), dtype=tf.string, name='parent')
text_len_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name='text_len')
parent_len_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name='parent_len')
author_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='author')
subreddit_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='subreddit')

text_layers = text_preprocessor(text_input)
text_layers = text_backbone(text_layers)['sequence_output']
parent_layers = parent_preprocessor(parent_input)
parent_layers = parent_backbone(parent_layers)['sequence_output']

text_layers = keras_nlp.layers.TransformerEncoder(num_heads=2, intermediate_dim=521, name="text.encoder"
                                                  )(text_layers)[:, text_backbone.cls_token_index, :]
parent_layers = keras_nlp.layers.TransformerEncoder(num_heads=2, intermediate_dim=521, name="parent.encoder"
                                                    )(parent_layers)[:, parent_backbone.cls_token_index, :]

text_layers = tf.keras.layers.Dense(10, name="text.encoded.pr")(text_layers)
parent_layers = tf.keras.layers.Dense(10, name="parent.encoded.pr")(parent_layers)
text_parent_layers = tf.keras.layers.Concatenate(name="text_parent.concat")([text_layers, parent_layers,
                                                                             text_len_input, parent_len_input])
text_parent_layers = tf.keras.layers.Dense(10, activation=activations.relu, name="text_parent.pr")(text_parent_layers)


author_layers = author_vectorizer(author_input)
subreddit_layers = subreddit_vectorizer(subreddit_input)
author_layers = tf.keras.layers.Dense(50, name="author.pr", activation=activations.relu)(author_layers)
subreddit_layers = tf.keras.layers.Dense(25, name="subreddit.pr", activation=activations.relu)(subreddit_layers)

contex_layers = tf.keras.layers.Concatenate(name="contex.concat")([author_layers, subreddit_layers])
contex_layers = tf.keras.layers.Dense(5, name="contex.pr", activation=activations.relu)(contex_layers)

global_layers = tf.keras.layers.Concatenate(name="global_concat")([text_parent_layers, contex_layers])
global_layers = tf.keras.layers.Dense(10, name="global.pr", activation=activations.relu)(global_layers)
output = tf.keras.layers.Dense(1, activation=activations.sigmoid, name='output')(global_layers)

model = keras.Model([text_input, parent_input, text_len_input, parent_len_input, author_input, subreddit_input], output)
model.compile(optimizer="adam", loss=losses.BinaryCrossentropy(), metrics=[metrics.BinaryAccuracy()])
model.summary()

tf.keras.utils.plot_model(model, to_file='model.png')
columns_order = ['text', 'parent', 'text_len', 'parent_len', 'author', 'subreddit']

model.fit(x=[df_train[col] for col in columns_order], y=df_train['sarcastic'], epochs=3,
          validation_data=([df_val[col] for col in columns_order], df_val['sarcastic']))
