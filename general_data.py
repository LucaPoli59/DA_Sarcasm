import constants

import os
import pandas as pd


df_full = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "data_full_sample.csv")).iloc[:, 1:].sample(frac=0.1)

df_train = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train.csv"), index_col="index").sample(frac=0.1)
df_train['date'] = pd.to_datetime(df_train['date'])

df_train_processed = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train_processed.csv"), index_col="index").sample(frac=0.1)
df_train_processed['date'] = pd.to_datetime(df_train_processed['date'])

df_val_processed = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "val_processed.csv"), index_col="index").sample(frac=0.1)
df_val_processed['date'] = pd.to_datetime(df_val_processed['date'])

df_test_processed = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "test_processed.csv"), index_col="index").sample(frac=0.1)
df_test_processed['date'] = pd.to_datetime(df_test_processed['date'])

target_info_rate = df_train['sarcastic'].value_counts(normalize=True).max()
#
# df_full = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "data_full_sample.csv")).iloc[:, 1:]
#
# df_train = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train.csv"), index_col="index")
# df_train['date'] = pd.to_datetime(df_train['date'])
#
# df_train_processed = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "train_processed.csv"), index_col="index")
# df_train_processed['date'] = pd.to_datetime(df_train_processed['date'])
#
# df_val_processed = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "val_processed.csv"), index_col="index")
# df_val_processed['date'] = pd.to_datetime(df_val_processed['date'])
#
# df_test_processed = pd.read_csv(os.path.join(constants.DATA_OUT_PATH, "test_processed.csv"), index_col="index")
# df_test_processed['date'] = pd.to_datetime(df_test_processed['date'])
#
# target_info_rate = df_train['sarcastic'].value_counts(normalize=True).max()


