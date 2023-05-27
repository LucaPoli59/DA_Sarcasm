from dash import html, dash_table, dcc, callback, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import timeit
from wordcloud import WordCloud

import constants
from general_data import df_train, target_info_rate, color_scale

# dfs_sp = {'author': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "author.csv"), index_col="element"),
#           'parent': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "parent.csv"), index_col="element"),
#           'date': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "date.csv"), index_col="element"),
#           'subreddit': pd.read_csv(os.path.join(constants.DATA_SP_PATH, "subreddit.csv"), index_col="element")}


df = pd.read_csv(os.path.join(constants.DATA_SP_PATH, "text_text_tokenized.csv"), index_col="element")
df['prop'] = round(df['prop'] * 100, 2)
df['info_rate'] = abs(df['prop'] - target_info_rate * 100)
num_words = df['tot'].sum()
df['tot_n'] = round(df['tot'] / num_words * 100, 2)


color_map = df['info_rate'].to_frame().reset_index().sort_values(by='info_rate')
info_min, info_max = color_map['info_rate'].min(), color_map['info_rate'].max()
color_map['rate_s'] = (color_map['info_rate'] - info_min) / (info_max - info_min)
color_map['color'] = px.colors.sample_colorscale(color_scale, color_map['rate_s'])

color_dict = color_map.set_index('element')['color'].to_dict()

end = timeit.default_timer()


word_cloud = WordCloud(width=1600, height=800, background_color='white',
                       color_func=lambda *args, **kwargs: color_dict[args[0]]
                       ).generate_from_frequencies(df['tot'].to_dict())

word_cloud_img = word_cloud.to_array()
df = df.reset_index()



bar_plot = px.bar(df, x='element', y='tot_n', color='info_rate', color_continuous_scale=color_scale,
                            range_x=[-0.3, 19.3], range_y=[0, df['tot_n'].max() + 0.1], hover_name='element',
                            hover_data={'tot': True, 'tot_n': False, 'info_rate': True, 'element': False, 'prop': True},
                            labels={'element': 'Parola', 'tot_n': 'Numero di campioni (%)', 'prop': 'Sarcastica (%)',
                                    'tot': 'Numero di campioni', 'info_rate': 'Rateo informativo'})




layout = dbc.Container(className="fluid", children=[
    html.Center(html.H1("Text Analysis", className="display-3 my-4")),
    dcc.Graph(figure=px.imshow(word_cloud_img).update_layout(margin={"t": 20, "b": 0, "r": 0, "l": 0, "pad": 0},
                                                             xaxis={"visible": False}, yaxis={"visible": False},
                                                             hovermode=False, coloraxis_showscale=True)),
    dcc.Graph(figure=bar_plot),

    dcc.Graph(figure=px.histogram(df, x='tot', y='info_rate', histfunc='avg', nbins=100, marginal='rug',
                                  hover_data=['tot', 'tot_n', 'info_rate'],
                                  labels={'element': 'Parola', 'tot': 'Numero di campioni',
                                          'tot_n': 'Numero di campioni (%)', 'info_rate': 'Rateo informativo'}
                                  )),

])
