from dash import html, dash_table, dcc, callback, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import os
import constants


df = pd.read_csv(os.path.join(constants.DATA_PATH, "train.csv"), index_col="index")

df_len = df[['sarcastic', 'text', 'parent']].copy()
df_len[['text', 'parent']] = df_len[['text', 'parent']].applymap(lambda x: len(x.split()))

text_range = min(df_len['text']), max(df_len['text'])
parent_range = min(df_len['parent']), max(df_len['parent'])

layout = dbc.Container(className="fluid", children=[
    dbc.Row(html.Center(html.H1("Dashboard", className="display-3 my-4"))),
    dbc.Row([
        dbc.Col(html.Div([
            html.H3("Informazioni generali sul dataset di train"),
            html.P("Numero di righe: " + str(df.shape[0])),
            html.P("Numero di righe nulle: " + str(df.isna().sum().sum())),
            html.P("Numero di righe duplicate: " + str(df.duplicated().sum())),
            html.Hr(),
            html.H5("Colonne e tipo")] +
            [html.P(col + ": " + str(df[col].dtype), style={'margin-bottom': '2px'}) for col in df.columns]
        )),
        dbc.Col(dcc.Graph(figure=px.histogram(df['sarcastic'],
                                              text_auto=True, histnorm='percent').update_layout(bargap=0.2)))
    ]),
    html.Hr(className="my-5"),
    dbc.Row(dash_table.DataTable(data=df.to_dict('records'), page_size=3,
                                 style_data={'whiteSpace': 'normal', 'height': 'auto'})),
    html.Hr(className="my-5"),
    dbc.Row(dcc.Graph(id="len_graph")),
    dcc.RangeSlider(id="len_slider", min=text_range[0], max=text_range[1], step=1, value=text_range)

])

@callback(Output(component_id='len_graph', component_property='figure'),
          [Input(component_id='len_slider', component_property='value')])
def update_len_graph(value):
    return px.histogram(df_len.loc[df_len['text'].between(*value)], x="text", color="sarcastic", histnorm='percent',
                                          text_auto=True, log_x=True)