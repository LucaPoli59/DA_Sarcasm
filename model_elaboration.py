import constants
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import activations, losses, metrics, callbacks
import keras_nlp


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


def get_hot_text_vectorizer(max_tokens=None, feature_name=None, vocabulary=None,
                            **kwargs):
    return tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        name=str(str(feature_name) + ".hot_encoding"),
        vocabulary=vocabulary,
        output_mode='multi_hot',
        split=None,
        standardize=None,
        **kwargs
    )


tmp_vectorizer = get_hot_text_vectorizer(max_tokens=10000)
tmp_vectorizer.adapt(df_train['author'])
author_vocab = tmp_vectorizer.get_vocabulary()

tmp_vectorizer = get_hot_text_vectorizer(max_tokens=1000)
tmp_vectorizer.adapt(df_train['subreddit'])
subreddit_vocab = tmp_vectorizer.get_vocabulary()

author_vectorizer = get_hot_text_vectorizer(max_tokens=10000, feature_name='author', vocabulary=author_vocab)
subreddit_vectorizer = get_hot_text_vectorizer(max_tokens=1000, feature_name='subreddit', vocabulary=subreddit_vocab)


bert_backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
bert_processor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
bert_backbone.trainable = False

text_input = tf.keras.Input(shape=(), dtype=tf.string, name='text')
parent_input = tf.keras.Input(shape=(), dtype=tf.string, name='parent')
text_len_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name='text_len')
parent_len_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name='parent_len')
author_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='author')
subreddit_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='subreddit')

text_parent_layers = bert_processor([text_input, parent_input])
text_parent_layers = bert_backbone(text_parent_layers)['sequence_output']
text_parent_layers = keras_nlp.layers.TransformerEncoder(num_heads=2,
                                                         intermediate_dim=521, name="text_parent.transformer"
                                                         )(text_parent_layers)[:, bert_backbone.cls_token_index, :]
text_parent_layers = tf.keras.layers.Dense(20, name="text_parent.pr1")(text_parent_layers)
text_parent_layers = tf.keras.layers.Concatenate(name="text_parent.concat")([text_parent_layers,
                                                                             text_len_input, parent_len_input])
text_parent_layers = tf.keras.layers.Dense(10, activation=activations.relu, name="text_parent.pr2")(text_parent_layers)

author_layers = author_vectorizer(author_input)
subreddit_layers = subreddit_vectorizer(subreddit_input)
author_layers = tf.keras.layers.Dense(50, name="author.pr", activation=activations.relu)(author_layers)
subreddit_layers = tf.keras.layers.Dense(25, name="subreddit.pr", activation=activations.relu)(subreddit_layers)

contex_layers = tf.keras.layers.Concatenate(name="contex.concat")([author_layers, subreddit_layers])
contex_layers = tf.keras.layers.Dense(5, name="contex.pr", activation=activations.relu)(contex_layers)

global_layers = tf.keras.layers.Concatenate(name="global.concat")([text_parent_layers, contex_layers])
global_layers = tf.keras.layers.Dense(10, name="global.pr", activation=activations.relu)(global_layers)
output = tf.keras.layers.Dense(1, activation=activations.sigmoid, name='output')(global_layers)

model = tf.keras.models.Model([text_input, parent_input, text_len_input, parent_len_input,
                               author_input, subreddit_input], output)
model.compile(optimizer="adam", loss=losses.BinaryCrossentropy(), metrics=[metrics.BinaryAccuracy()])
model.summary()

tf.keras.utils.plot_model(model, to_file='model.png')
columns_order = ['text', 'parent', 'text_len', 'parent_len', 'author', 'subreddit']

epochs = 1
history = model.fit(x=[df_train[col] for col in columns_order], y=df_train['sarcastic'], epochs=epochs, batch_size=128,
                    callbacks=[callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=int(epochs/3),
                                                       restore_best_weights=True)],
                    validation_data=([df_val[col] for col in columns_order], df_val['sarcastic']))


history = pd.DataFrame(history.history)
history = history.rename(columns={'binary_accuracy': 'accuracy', 'val_binary_accuracy': 'val_accuracy'})
history.index.name = "epoch"
history.to_csv(os.path.join(constants.MODEL_DIR, "history.csv"))

model.save(os.path.join(constants.MODEL_DIR, "model.h5"))


