import pandas as pd
import os
import constants

df_full = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "data_full.csv")).iloc[:, 1:]
df_train = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train.csv"), index_col="index")
df_train_processed = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train_processed.csv"), index_col="index")
df_val_processed = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "val_processed.csv"), index_col="index")
df_test_processed = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "test_processed.csv"), index_col="index")

df_train['date'] = pd.to_datetime(df_train['date'])
df_train_processed['date'] = pd.to_datetime(df_train_processed['date'])
df_val_processed['date'] = pd.to_datetime(df_val_processed['date'])
df_test_processed['date'] = pd.to_datetime(df_test_processed['date'])

df_full_stats = pd.Series(dtype=object)
df_full_stats['len'] = df_full.shape[0]
df_full_stats['null_values'] = df_full.isna().sum().sum()
df_full_stats['duplicated'] = df_full.duplicated().sum()
df_full_stats_types = df_full.dtypes

df_train_stats = pd.Series(dtype=object)
df_train_stats['len'] = df_train.shape[0]
df_train_stats['null_values'] = df_train.isna().sum().sum()
df_train_stats['duplicated'] = df_train.duplicated().sum()
df_train_stats_types = df_train.dtypes
df_train_stats_target = df_train['sarcastic'].value_counts(normalize=True)

df_full_stats.to_csv(os.path.join(constants.APP_DATA_DIR, "df_full_stats.csv"))
df_full_stats_types.to_csv(os.path.join(constants.APP_DATA_DIR, "df_full_stats_types.csv"))
df_train_stats.to_csv(os.path.join(constants.APP_DATA_DIR, "df_train_stats.csv"))
df_train_stats_types.to_csv(os.path.join(constants.APP_DATA_DIR, "df_train_stats_types.csv"))
df_train_stats_target.to_csv(os.path.join(constants.APP_DATA_DIR, "df_train_stats_target.csv"))

df_train.sample(constants.SAMPLE_DIM).to_csv(os.path.join(constants.APP_DATA_DIR, "df_train.csv"))
df_train_processed.sample(constants.SAMPLE_DIM).to_csv(os.path.join(constants.APP_DATA_DIR, "df_train_processed.csv"))
df_val_processed.sample(constants.SAMPLE_DIM).to_csv(os.path.join(constants.APP_DATA_DIR, "df_val_processed.csv"))
df_test_processed.sample(constants.SAMPLE_DIM).to_csv(os.path.join(constants.APP_DATA_DIR, "df_test_processed.csv"))