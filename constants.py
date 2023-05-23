import os


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

ENABLE_OUT = False
TARGET = 'sarcastic'
CONTEXT_COLS = ['author', 'subreddit', 'parent']
EMBEDDING_DIM = 300