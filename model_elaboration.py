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

bert_backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
bert_processor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
bert_backbone.trainable = False

text_input = tf.keras.Input(shape=(), dtype=tf.string, name='text')
parent_input = tf.keras.Input(shape=(), dtype=tf.string, name='parent')

text_parent_layers = bert_processor([text_input, parent_input])
text_parent_layers = bert_backbone(text_parent_layers)['sequence_output']
text_parent_layers = keras_nlp.layers.TransformerEncoder(num_heads=2, intermediate_dim=521, name="text_parent.encoder"
                                                         )(text_parent_layers)[:, bert_backbone.cls_token_index, :]
text_parent_layers = tf.keras.layers.Dense(10, name="text_parent.encoded.pr")(text_parent_layers)
# parent_layers = tf.keras.layers.Dense(10, name="parent.encoded.pr")(parent_layers)
# text_parent_layers = tf.keras.layers.Concatenate(name="text_parent.concat")([text_layers, parent_layers,
#                                                                              text_len_input, parent_len_input])
# text_parent_layers = tf.keras.layers.Dense(10, activation=activations.relu, name="text_parent.pr")(text_parent_layers)

# global_layers = tf.keras.layers.Dense(10, name="global.pr", activation=activations.relu)(text_parent_layers)
output = tf.keras.layers.Dense(1, activation=activations.sigmoid, name='output')(text_parent_layers)

model = keras.Model([text_input, parent_input], output)
model.compile(optimizer="adam", loss=losses.BinaryCrossentropy(), metrics=[metrics.BinaryAccuracy()])
model.summary()

tf.keras.utils.plot_model(model, to_file='model.png')
# columns_order = ['text', 'parent', 'text_len', 'parent_len', 'author', 'subreddit']
columns_order = ['text', 'parent']

model.fit(x=[df_train[col] for col in columns_order], y=df_train['sarcastic'], epochs=3,
          validation_data=([df_val[col] for col in columns_order], df_val['sarcastic']))
