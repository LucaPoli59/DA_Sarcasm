import dash
from dash import html, dcc
import plotly.express as px
import dash_bootstrap_components as dbc
import constants
import os
import general_data as gd
import pandas as pd
import tensorflow as tf
import keras_nlp

# train_history = pd.read_csv(os.path.join(constants.MODEL_DIR, "history.csv"))
#
# # todo: Spostare queste righe di valuation sul val nel file di elaborazione del modello
# df_val = gd.df_val_processed.sample(frac=0.01)
# df_val[df_val.columns[1:]] = df_val[df_val.columns[1:]].astype(str)
# df_val['text_len'] = df_val['text'].apply(lambda x: len(x.split()))
# df_val['text_len'] = df_val['text_len'] / df_val['text_len'].max()
# df_val['parent_len'] = df_val['parent'].apply(lambda x: len(x.split()))
# df_val['parent_len'] = df_val['parent_len'] / df_val['parent_len'].max()
#
#
model = tf.keras.models.load_model(os.path.join(constants.MODEL_DIR, "model.h5"))
# compare_df = pd.DataFrame(columns=['True', 'Predicted'])
# compare_df['real'] = df_val['sarcastic']
# compare_df['predicted'] = model.predict([df_val[col] for col in constants.MODEL_COLUMNS_ORDER]
#                                         ).round().astype(bool).reshape(-1)
#
print("model training loaded")
#
# layout = dbc.Container(className="fluid", children=[
#     html.Center(html.H1("Addestramento del modello", className="display-3 my-4")),
#     html.Center(html.H3("Immagine del modello", className="my-4")),
#     html.Img(src=dash.get_asset_url("model_img.png"), className="img-fluid"),
#     html.Img(src=dash.get_asset_url("model_img_bert.png"), className="img-fluid mt-3"),
#
#     html.Hr(className="my-5"),
#     html.Center(html.H3("History di training del modello")),
#     dbc.Row([
#         dbc.Col(dcc.Graph(figure=px.line(train_history, x="epoch", y=["loss", "val_loss"]))),
#         dbc.Col(dcc.Graph(figure=px.line(train_history, x="epoch", y=["accuracy", "val_accuracy"])))
#     ]),
#
#     html.Hr(className="my-5"),
#     html.Center(html.H3("Risultati sul dataset di validation")),
# ])

layout = ""