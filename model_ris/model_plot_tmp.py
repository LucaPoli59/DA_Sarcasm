import constants
import tensorflow as tf
import keras_nlp
import general_data as gdf
import os
import plotly.express as px


model = tf.keras.models.load_model(os.path.join(gdf.constants.MODEL_DIR, "model.h5"))

tf.keras.utils.plot_model(model, to_file=os.path.join(gdf.constants.ASSET_DIR, "model_img.png"), rankdir="LR")
tf.keras.utils.plot_model(model, to_file=os.path.join(gdf.constants.ASSET_DIR, "model_img_bert.png"), rankdir="LR",
                          expand_nested=True, layer_range=['text', 'text_parent.transformer'])

