from dash import html, dash_table, dcc, callback, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import constants
import timeit

dfs_sp = {'author': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "author.csv"), index_col="element"),
          'parent': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "parent.csv"), index_col="element"),
          'date': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "date.csv"), index_col="element"),
          'subreddit': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "subreddit.csv"), index_col="element")}

df_tmp = dfs_sp['subreddit']

tot_min, tot_max = df_tmp['tot'].describe()[['min', 'max']]
bins = np.floor(np.logspace(np.log10(tot_min), np.log10(tot_max+1), 11))

df_tmp = df_tmp.groupby(pd.cut(df_tmp['tot'], bins=bins, labels=False, include_lowest=True)).sum().cumsum()
df_tmp['prop'] = df_tmp['sarc_freq'] / df_tmp['tot']

print(df_tmp)
print(bins)


layout = dbc.Container(className="fluid", children=[
    # html.Center(html.H1("Context Analysis", className="display-3 my-4")),
    # dcc.Graph(figure=px.histogram(df_t['sarcastic'], text_auto=True, histnorm='percent'
    #                               ).update_layout(bargap=0.2, title_text="Analisi feature uniche", title_x=0.5)),
    # html.Hr(className="my-5"),
    # dbc.Container(className="d-flex flex-column justify-content-center align-items-center", children=[
    #     html.Center(html.H3(id='len_title', children="Distribuzione della lunghezza del testo")),
    #     dbc.RadioItems(id="feature_len_selector", options={'text': 'Testo', 'parent': 'Parent'}, value='text',
    #                    inline=True, className="date-group-items justify-content-center mt-3"),
    # ]),
    # dcc.Graph(id="sarcastic_proportion_graph"),
    # dbc.Container(className="mt-3", children=[
    #     dbc.Label("Numero di campioni per gruppo:"),
    #     dcc.RangeSlider(id="size_slider", min=0, max=100, step=1, dots=False, value=(0, 50), className="mt-1",
    #                     allowCross=False),
    #
    # ]),
])

#
# @callback([Output(component_id='len_slider', component_property='marks'),
#            Output(component_id='len_slider', component_property='value'),
#            Output(component_id='tot_slider', component_property='marks'),
#            Output(component_id='tot_slider', component_property='value'),
#            Output(component_id='len_title', component_property='children')],
#           [Input(component_id='feature_len_selector', component_property='value')])
# def update_sliders(ft_s):
#     len_slider_marks = {mark: str(round(v)) for mark, v in len_range[ft_s].iloc[::20].items()}
#     tot_slider_marks = {mark: str(round(v)) for mark, v in tot_range[ft_s].iloc[::10].items()}
#
#     title = "Distribuzione della lunghezza del "
#     if ft_s == 'text':
#         title += "testo"
#     else:
#         title += "parent"
#
#     return len_slider_marks, (0, 50), tot_slider_marks, (0, 50), title
#
#
# @callback(Output(component_id='len_graph', component_property='figure'),
#           [Input(component_id='feature_len_selector', component_property='value'),
#            Input(component_id='len_slider', component_property='value'),
#            Input(component_id='tot_slider', component_property='value')])
# def update_len_graph(ft_s, len_value, tot_value):
#     df_len = len_dfs[ft_s]
#     len_value = len_range[ft_s][len_value[0]], len_range[ft_s][len_value[1]]
#     tot_value = (tot_range[ft_s][tot_value[0]], tot_range[ft_s][tot_value[1]])[::-1]
#
#     return px.bar(df_len.loc[df_len['len'].between(*len_value) & df_len['tot'].between(*tot_value)],
#                   x="len", y="prop_s", text_auto=True, hover_data=['prop', 'tot'], range_y=[-0.51, 0.51],
#                   labels={'len': 'Numero parole', 'prop': 'Sarcastica (%)',
#                           'prop_s': 'Rateo informativo', 'tot': 'Numero campioni'})

