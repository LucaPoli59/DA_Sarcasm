import os
import constants
import keras_nlp
import pandas as pd
import tensorflow as tf



def open_dataframe(name, text_max=None, parent_max=None):
    df = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, name + ".csv"), index_col='index')
    df['sarcastic'] = df['sarcastic'].astype(int)
    df[df.columns[1:]] = df[df.columns[1:]].astype(str)

    df['text_len'] = df['text'].apply(lambda x: len(x.split()))
    df['parent_len'] = df['parent'].apply(lambda x: len(x.split()))

    if text_max is None:
        text_max = df['text_len'].max()
    if parent_max is None:
        parent_max = df['parent_len'].max()

    df['text_len'] = df['text_len'] / text_max
    df['parent_len'] = df['parent_len'] / parent_max
    return df, text_max, parent_max


# Carico i dati
df_train, text_len_max, parent_len_max = open_dataframe("train_processed")
df_val = open_dataframe("val_processed", text_max=text_len_max, parent_max=parent_len_max)[0]
df_test = open_dataframe("test_processed", text_max=text_len_max, parent_max=parent_len_max)[0]

len_max = pd.Series({'text': text_len_max, 'parent': parent_len_max})
len_max.to_csv(os.path.join(constants.MODEL_DIR, "len_max.csv"))

# Se il modello deve essere definito lo vado a creare, altrimenti lo carico da file
if not constants.LOAD_MODEL:
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


    subreddit_tokens = 4000

    tmp_vectorizer = get_hot_text_vectorizer(max_tokens=subreddit_tokens)
    tmp_vectorizer.adapt(df_train['subreddit'])
    subreddit_vocab = tmp_vectorizer.get_vocabulary()
    subreddit_vectorizer = get_hot_text_vectorizer(max_tokens=subreddit_tokens, feature_name='subreddit',
                                                   vocabulary=subreddit_vocab)

    bert_backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
    bert_processor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
    bert_backbone.trainable = False

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    parent_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='parent')
    text_len_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='text_len')
    parent_len_input = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='parent_len')
    subreddit_input = tf.keras.layers.Input(shape=(1,), dtype=tf.string, name='subreddit')

    text_parent_layers = bert_processor([text_input, parent_input])
    text_parent_layers = bert_backbone(text_parent_layers)['sequence_output']
    text_parent_layers = keras_nlp.layers.TransformerEncoder(num_heads=4, dropout=0.1,
                                                             intermediate_dim=250, name="text_parent.transformer"
                                                             )(text_parent_layers)[:, bert_backbone.cls_token_index, :]
    text_parent_layers = tf.keras.layers.Concatenate(name="text_parent.concat")([text_parent_layers,
                                                                                 text_len_input, parent_len_input])
    text_parent_layers = tf.keras.layers.Dense(100, activation=tf.keras.activations.relu,
                                               name="text_parent.pr")(text_parent_layers)
    text_parent_layers = tf.keras.layers.Dropout(0.1)(text_parent_layers)

    subreddit_layers = subreddit_vectorizer(subreddit_input)
    subreddit_layers = tf.keras.layers.Dense(25, name="subreddit.pr",
                                             activation=tf.keras.activations.relu)(subreddit_layers)
    subreddit_layers = tf.keras.layers.Dropout(0.1)(subreddit_layers)

    global_layers = tf.keras.layers.Concatenate(name="global.concat")([text_parent_layers, subreddit_layers])
    global_layers = tf.keras.layers.Dense(20, name="global.pr",
                                          activation=tf.keras.activations.relu)(global_layers)
    output = tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid, name='output')(global_layers)

    model = tf.keras.models.Model([text_input, parent_input, text_len_input, parent_len_input, subreddit_input], output)
    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    model.summary()

    epochs = 50
    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(epochs / 5),
                                                 restore_best_weights=True),
                tf.keras.callbacks.BackupAndRestore(backup_dir=os.path.join(constants.MODEL_DIR, "backup"),
                                                    delete_checkpoint=False),
                tf.keras.callbacks.CSVLogger(os.path.join(constants.MODEL_DIR, "history.csv"), append=True)]

    history = model.fit(x=[df_train[col] for col in constants.MODEL_COLUMNS_ORDER], y=df_train['sarcastic'],
                        epochs=epochs, batch_size=128, callbacks=callback, shuffle=True,
                        validation_data=([df_val[col] for col in constants.MODEL_COLUMNS_ORDER], df_val['sarcastic']))

    model.save(os.path.join(constants.MODEL_DIR, "model.h5"))


    def create_compare_df(df):
        compare_df = pd.DataFrame(columns=['True', 'Predicted'])
        compare_df['True'] = df['sarcastic']
        compare_df['score'] = model.predict([df[col] for col in constants.MODEL_COLUMNS_ORDER]).reshape(-1)
        compare_df['Predicted'] = compare_df['score'].values.round().astype(bool).reshape(-1)

        compare_df[['True', 'Predicted']] = compare_df[['True', 'Predicted']].applymap(
            lambda x: "Sarcastico" if x == 1 else "Non sarcastico")

        return compare_df


    cmp_train, cmp_val, cmp_test = create_compare_df(df_train), create_compare_df(df_val), create_compare_df(df_test)

    cmp_val.to_csv(os.path.join(constants.MODEL_DIR, "compare_val.csv"))
    cmp_test.to_csv(os.path.join(constants.MODEL_DIR, "compare_test.csv"))

    tf.keras.utils.plot_model(model, to_file=os.path.join(constants.ASSET_DIR, "model_img.png"), rankdir="LR")
    tf.keras.utils.plot_model(model, to_file=os.path.join(constants.ASSET_DIR, "model_img_bert.png"), rankdir="LR",
                              expand_nested=True, layer_range=['text', 'text_parent.transformer'])

else:
    model = tf.keras.models.load_model(os.path.join(constants.MODEL_DIR, "model.h5"))
    cmp_val = pd.read_csv(os.path.join(constants.MODEL_DIR, "compare_val.csv"), index_col=0)
    cmp_test = pd.read_csv(os.path.join(constants.MODEL_DIR, "compare_test.csv"), index_col=0)

