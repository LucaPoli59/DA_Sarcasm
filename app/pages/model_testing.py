import datetime as dt
import os
import constants
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import keras_nlp
import tensorflow as tf
from dash import callback, Output, Input, State
from dash import html, dcc, dash_table
from nltk import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

import general_data as gdf

model = tf.keras.models.load_model(os.path.join(constants.MODEL_DIR, "model.h5"))
date_min = gdf.df_test_processed['date'].min()


cmp_test = pd.read_csv(os.path.join(constants.MODEL_DIR, "compare_test.csv"), index_col=0)
len_max = pd.read_csv(os.path.join(constants.MODEL_DIR, "len_max.csv"), index_col=0).iloc[:, 0]
del_punctuation = pd.read_csv(os.path.join(constants.MODEL_DIR, "del_punctuation.csv"), index_col=0)

test_fpr, test_tpr, test_thresholds = roc_curve(cmp_test['True'], cmp_test['score'])
test_roc = pd.DataFrame({'fpr': test_fpr, 'tpr': test_tpr}, index=pd.Index(test_thresholds, name='thresholds'))
test_roc.columns.name = 'Rateo'
test_auc = auc(test_fpr, test_tpr)
test_roc_curve = px.area(x=test_roc['fpr'], y=test_roc['tpr'],
                         labels={'x': 'Rateo falsi positivi', 'y': 'Rateo veri positivi'})
test_roc_curve = test_roc_curve.add_shape(type='line', line={'dash': 'dash'}, x0=0, x1=1, y0=0, y1=1)
test_roc_curve = test_roc_curve.update_yaxes(
    scaleanchor="x", scaleratio=1).update_xaxes(constrain='domain').update_layout(title_text=f'AUC = {test_auc:.4f}',
                                                                                  title_x=0.5)

test_rp = classification_report(cmp_test['True'], cmp_test['Predicted'], target_names=['Non sarcastico', 'Sarcastico'],
                                output_dict=True)
test_rp_acc = round(test_rp.pop('accuracy'), 3)
test_rp = pd.DataFrame(test_rp).loc[['precision', 'recall', 'f1-score'], ['Non sarcastico', 'Sarcastico']].transpose()
test_rp = test_rp.round(3).reset_index().rename(columns={'index': 'Classe'})

layout = dbc.Container(className="fluid", children=[
    html.Center(html.H1("Demo del modello addestrato", className="display-3 my-4")),
    dbc.Container(className="border border-secondary mt-5", children=[
        html.Center(html.H3("Demo del modello", className="my-4")),

        dbc.InputGroup([
            dbc.InputGroupText("Testo", className="fs-5"),
            dbc.Input(id="demo_input_text", type="text", debounce=True, className="form-control fs-5"),
            html.Span(dbc.Button(html.I(className="bi bi-dice-5 fs-4"), outline=True,
                                 id="demo_input_text_random"),
                      className="input-group-text"),
        ], className="mb-3"),
        dbc.InputGroup([
            dbc.InputGroupText("Parent", className="fs-5"),
            dbc.Input(id="demo_input_parent", type="text", debounce=True, className="form-control fs-5"),
            html.Span(dbc.Button(html.I(className="bi bi-dice-5 fs-4"), outline=True,
                                 id="demo_input_parent_random"),
                      className="input-group-text"),
        ], className="mb-3"),

        dbc.Row([
            dbc.InputGroup([
                dbc.InputGroupText("Subreddit", className="fs-5"),
                dbc.Input(id="demo_input_subreddit", type="text", debounce=True,
                          className="form-control fs-5"),
                html.Span(dbc.Button(html.I(className="bi bi-dice-5 fs-4"), outline=True,
                                     id="demo_input_subreddit_random"),
                          className="input-group-text"),
            ], className="col"),
            dbc.InputGroup([
                dbc.InputGroupText("Autore", className="fs-5"),
                dbc.Input(id="demo_input_author", type="text", debounce=True, className="form-control fs-5"),
                html.Span(dbc.Button(html.I(className="bi bi-dice-5 fs-4"), outline=True,
                                     id="demo_input_author_random"),
                          className="input-group-text"),
            ], className="col")
        ], className="mb-3"),

        dbc.Row([
            dbc.InputGroup([
                dbc.InputGroupText("Data", className="fs-5"),
                dcc.DatePickerSingle(id="demo_input_date", min_date_allowed=dt.date(2000, 1, 1),
                                     max_date_allowed=dt.date.today(),
                                     initial_visible_month=date_min.date(),
                                     className="input-group-text fs-5"),
                html.Span(dbc.Button(html.I(className="bi bi-dice-5 fs-4"), outline=True,
                                     id="demo_input_date_random"),
                          className="input-group-text"),
            ], className="col"),

            dbc.Col([
                dbc.Button("Classifica il commento", id="demo_button_predict",
                           color="primary", size="lg", className="fs-5 mt-2"),
            ], className="col"),
        ], className="mb-3"),

        html.Div(className="my-3 d-inline-flex gap-3", children=[
            html.Label("Il commento è:", className="fs-5"),
            dcc.Loading(html.Div(id="demo_output_prediction", className="fs-5", children="       "))
        ])
    ]),
    html.Center(html.H3("Risultati sul dataset di test", className="my-5")),
    dbc.Row([
        dbc.Col([
            html.Center(html.H5("Confusion Matrix")),
            dcc.Graph(figure=px.imshow(confusion_matrix(cmp_test['True'], cmp_test['Predicted'], normalize='true'),
                                       x=['Non Sarcastico', 'Sarcastico'], y=['Non Sarcastico ', 'Sarcastico '],
                                       text_auto=True, color_continuous_scale='blues',
                                       labels={'color': 'Percentuale', 'x': 'Classe predetta', 'y': 'Classe reale'}
                                       ).update_layout(coloraxis_showscale=False))
        ], className="col-6"),
        dbc.Col([
            dbc.Row(html.Center(html.H5("Report della classificazione"))),
            dbc.Row(html.P("Accuracy: " + str(test_rp_acc))),
            dbc.Row(dash_table.DataTable(test_rp.to_dict('records'))),

        ], className="col-6"),
    ], className="d-flex align-items-center"),
    html.Center(html.H5("Confronto tra la probabilità predetta e la classe reale")),
    dcc.Graph(figure=px.histogram(cmp_test, x='score', color="True",
                                  labels={'score': 'Probabilità predetta', 'True': 'Classe reale',
                                          'count': 'Numero di istanze', 0: 'Non sarcastico', 1: 'Sarcastico'})),
    dbc.Row([
        dbc.Col([
            html.Center(html.H5("Falsi positivi e Veri positivi ad ogni soglia")),
            dcc.Graph(figure=px.line(test_roc, labels={'fpr': 'Rateo falsi positivi', 'tpr': 'Rateo veri positivi',
                                                       'thresholds': 'soglia'}
                                     ).update_yaxes(scaleanchor="x", scaleratio=1
                                                    ).update_xaxes(range=[0, 1], constrain='domain'))
        ], className="col-6"),
        dbc.Col([
            html.Center(html.H5("Curva ROC")),
            dcc.Graph(figure=test_roc_curve)

        ], className="col-6")
    ])

])


def _process_text(text):
    tokenizer = TweetTokenizer()

    text = tokenizer.tokenize(text.lower())
    text = [word for word in text if word not in del_punctuation.values]
    text = [word for word in text if word not in stopwords.words('english')]
    return " ".join(text)


@callback(Output("demo_input_text", "value"), [Input("demo_input_text_random", "n_clicks")])
def random_text(n_clicks):
    return gdf.df_test_processed.sample(1)['text'].values[0]


@callback(Output("demo_input_parent", "value"), [Input("demo_input_parent_random", "n_clicks")])
def random_parent(n_clicks):
    return gdf.df_test_processed.sample(1)['parent'].values[0]


@callback(Output("demo_input_subreddit", "value"), [Input("demo_input_subreddit_random", "n_clicks")])
def random_subreddit(n_clicks):
    return gdf.df_test_processed.sample(1)['subreddit'].values[0]


@callback(Output("demo_input_author", "value"), [Input("demo_input_author_random", "n_clicks")])
def random_author(n_clicks):
    return gdf.df_test_processed.sample(1)['author'].values[0]


@callback(Output("demo_input_date", "date"), [Input("demo_input_date_random", "n_clicks")])
def random_date(n_clicks):
    return gdf.df_test_processed.sample(1)['date'].iloc[0].date()


@callback([Output("demo_output_prediction", "children"), Output("demo_output_prediction", "style")],
          [Input("demo_button_predict", "n_clicks")],
          [State("demo_input_text", "value"), State("demo_input_parent", "value"),
           State("demo_input_subreddit", "value"), State("demo_input_author", "value"),
           State("demo_input_date", "date")], prevent_initial_call=True)
def predict(n_clicks, text, parent, subreddit, author, date):
    if text == "" or parent == "" or subreddit == "" or author == "" or date == "":
        return "Inserire tutti i campi", {"color": "yellow"}
    else:

        text, parent = _process_text(text), _process_text(parent)
        text_len, parent_len = len(text.split()) / len_max['text'], len(parent.split()) / len_max['parent']
        instance = [text, parent, text_len, parent_len, subreddit]
        pred = model.predict(x=[np.array([val]) for val in instance], batch_size=1)[0][0]

        if round(pred) == 1:
            return "Sarcastico, con probabilità del " + str(round(pred * 100)) + "%", {"color": "green"}
        return "Non Sarcastico, con probabilità del " + str(round(pred * 100)) + "%", {"color": "red"}
