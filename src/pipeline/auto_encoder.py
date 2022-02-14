import pandas as pd
from tensorflow.keras.models import Model
import tensorflow as tf


class AutoEncoder(Model):
    def __init__(self,
                 encoder_layers: list,
                 decoder_layers: list):
        super(AutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential(layers=encoder_layers, name="encoder")
        self.decoder = tf.keras.Sequential(layers=decoder_layers, name="decoder")

    def call(self, x, **kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_normal_train_data(x, y):
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values.reshape(-1)
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.values
    y = y.astype(bool)
    return x[y], x[~y]
