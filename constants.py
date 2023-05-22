import os


PROJECT_NAME = "data_analytics"
PROJECT_PATH = os.getcwd()

while os.path.basename(os.getcwd()) != PROJECT_NAME:
    os.chdir("..")
    PROJECT_PATH = os.getcwd()

print(PROJECT_PATH)

DATA_PATH = os.path.join(PROJECT_PATH, "dataset")
GLOVE_PATH = os.path.join(DATA_PATH, "glove.json")