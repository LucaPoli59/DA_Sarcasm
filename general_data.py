import os

import pandas as pd

import constants

df_full_stats = pd.read_csv(os.path.join(constants.APP_DATA_DIR, "df_full_stats.csv"), index_col=0).iloc[:, 0]
df_full_stats_dt = pd.read_csv(os.path.join(constants.APP_DATA_DIR, "df_full_stats_types.csv"), index_col=0).iloc[:, 0]
df_train_stats = pd.read_csv(os.path.join(constants.APP_DATA_DIR, "df_train_stats.csv"), index_col=0).iloc[:, 0]
df_train_stats_dt = pd.read_csv(os.path.join(constants.APP_DATA_DIR, "df_train_stats_types.csv"),
                                index_col=0).iloc[:, 0]
target_distribution = pd.read_csv(os.path.join(constants.APP_DATA_DIR, "df_train_stats_target.csv"),
                                  index_col=0).iloc[:, 0]

df_train = pd.read_csv(os.path.join(constants.APP_DATA_DIR, "df_train.csv"), index_col=0)
df_train_processed = pd.read_csv(os.path.join(constants.APP_DATA_DIR, "df_train_processed.csv"), index_col=0)
df_val_processed = pd.read_csv(os.path.join(constants.APP_DATA_DIR, "df_val_processed.csv"), index_col=0)
df_test_processed = pd.read_csv(os.path.join(constants.APP_DATA_DIR, "df_test_processed.csv"), index_col=0)

df_train['date'] = pd.to_datetime(df_train['date'])
df_train_processed['date'] = pd.to_datetime(df_train_processed['date'])
df_val_processed['date'] = pd.to_datetime(df_val_processed['date'])
df_test_processed['date'] = pd.to_datetime(df_test_processed['date'])

target_info_rate = target_distribution.max()
