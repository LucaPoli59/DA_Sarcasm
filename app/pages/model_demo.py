from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import general_data as gdf
import datetime as dt
import os
import constants
import tensorflow as tf
import keras_nlp


# model = tf.keras.models.load_model(os.path.join(gdf.constants.MODEL_DIR, "model.h5"))
#
# print(model.summary())

date_min = gdf.df_test_processed['date'].min()

layout = dbc.Container([
    dbc.Row([
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
                html.Div(id="demo_output_prediction", className="fs-5")
            ])
        ]),
        html.Center(html.H3("Risultati sul dataset di test", className="my-5")),

    ])
])


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
        return "", {}
    else:
        #realizzare la predizione
        # return "Sarcastico", {"color": "green"}
        return "Non Sarcastico", {"color": "red"}