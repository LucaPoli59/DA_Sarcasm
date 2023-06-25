import os
import plotly.express as px
import silence_tensorflow.auto

PROJECT_NAME = "data_analytics"
PROJECT_PATH = os.getcwd()

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

ASSET_DIR = os.path.join(PROJECT_PATH, "app", "assets")
if not os.path.exists(ASSET_DIR):
    os.mkdir(ASSET_DIR)

ENABLE_OUT = False
TARGET = 'sarcastic'
CONTEXT_COLS = ['author', 'subreddit', 'parent']
MODEL_COLUMNS_ORDER = ['text', 'parent', 'text_len', 'parent_len', 'author', 'subreddit']


COLOR_SCALE = px.colors.sequential.Turbo_r



# da mettere come ultima riga del file di training
# shutil.copy2(os.path.join(constants.MODEL_DIR, "model.png"), os.path.join(constants.ASSETS_DIR, "model.png"))