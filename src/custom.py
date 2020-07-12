import tcn
import tensorflow as tf
from keras_adamw.optimizers_v2 import SGDW, AdamW
from tensorflow.keras.layers import Activation

from ind_rnn import IndRNNCell, IndRNN
from bistable import BRUCell, nBRUCell, BRU, nBRU

def swish(x):
    return tf.sigmoid(x) * x

custom_objects = tf.keras.utils.get_custom_objects()
custom_objects.update(
    {
        'TCN': tcn.TCN,
        'IndRNNCell': IndRNNCell,
        'BRUCell': BRUCell,
        'nBRUCell': nBRUCell,
        'AdamW': AdamW,
        'SGDW': SGDW,
        'swish': swish
    }
)