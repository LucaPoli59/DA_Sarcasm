import constants
import tensorflow as tf
import keras_nlp
import general_data as gdf
import os
import plotly.express as px
import pydotplus


model = tf.keras.models.load_model(os.path.join(gdf.constants.MODEL_DIR, "model.h5"))

tf.keras.utils.plot_model(model, to_file=os.path.join(gdf.constants.MODEL_DIR, "model_img_full.png"), expand_nested=True)
tf.keras.utils.plot_model(model, to_file=os.path.join(gdf.constants.MODEL_DIR, "model_img_v.png"), rankdir="TB")
tf.keras.utils.plot_model(model, to_file=os.path.join(gdf.constants.MODEL_DIR, "model_img_h.png"), rankdir="LR")
tf.keras.utils.plot_model(model, to_file=os.path.join(gdf.constants.MODEL_DIR, "model_img_bert_h.png"), rankdir="LR",
                          expand_nested=True, layer_range=['text', 'text_parent.transformer'])
tf.keras.utils.plot_model(model, to_file=os.path.join(gdf.constants.MODEL_DIR, "model_img_bert_v.png"), rankdir="TB",
                          expand_nested=True, layer_range=['text', 'text_parent.transformer'])

