import os

os.popen("pip install -r requirements.txt")

import nltk
import plotly.express as px

nltk.download('stopwords')

PROJECT_NAME = "data_analytics"
PROJECT_PATH = os.getcwd()
SAMPLE_DIM = 1000

while os.path.basename(os.getcwd()) != PROJECT_NAME:
    os.chdir("..")
    PROJECT_PATH = os.getcwd()


DATA_PATH = os.path.join(PROJECT_PATH, "dataset")
DATA_IN_PATH = os.path.join(DATA_PATH, "input")
GLOVE_PATH = os.path.join(DATA_PATH, "glove.json")

DATA_OUT_PATH = os.path.join(DATA_PATH, "output")
if not os.path.exists(DATA_OUT_PATH):
    os.mkdir(DATA_OUT_PATH)

DATA_SP_PATH = os.path.join(DATA_OUT_PATH, "sp")
if not os.path.exists(DATA_SP_PATH):
    os.mkdir(DATA_SP_PATH)

DATA_WC_PATH = os.path.join(DATA_SP_PATH, "wc")
if not os.path.exists(DATA_WC_PATH):
    os.mkdir(DATA_WC_PATH)

MODEL_DIR = os.path.join(PROJECT_PATH, "model_ris")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

MODEL_CMP_DIR = os.path.join(MODEL_DIR, "cmp_model")
if not os.path.exists(MODEL_CMP_DIR):
    os.mkdir(MODEL_CMP_DIR)

ASSET_DIR = os.path.join(PROJECT_PATH, "app", "assets")
if not os.path.exists(ASSET_DIR):
    os.mkdir(ASSET_DIR)

APP_DATA_DIR = os.path.join(PROJECT_PATH, "app", "data")
if not os.path.exists(APP_DATA_DIR):
    os.mkdir(APP_DATA_DIR)

LOAD_MODEL = True
TARGET = 'sarcastic'
MODEL_COLUMNS_ORDER = ['text', 'parent', 'text_len', 'parent_len', 'subreddit']


COLOR_SCALE = px.colors.sequential.Turbo_r
