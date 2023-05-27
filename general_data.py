import constants

import os
import pandas as pd
import plotly.express as px

df_full = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "data_full_sample.csv")).iloc[:, 1:]

df_train = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train.csv"), index_col="index")
df_train['date'] = pd.to_datetime(df_train['date'])

target_info_rate = df_train['sarcastic'].value_counts(normalize=True).max()

color_scale = px.colors.sequential.Turbo_r
