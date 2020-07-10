import tcn
import tensorflow as tf
from keras_adamw.optimizers_v2 import SGDW, AdamW
from tensorflow.keras.layers import Activation

from ind_rnn import IndRNN
from bistable import BRUCell, nBRUCell

def swish(x):
    return tf.sigmoid(x) * x

custom_objects = tf.keras.utils.get_custom_objects()
custom_objects.update(
    {
        'TCN': tcn.TCN,
        'IndRNN': IndRNN,
        'BRUCell': BRUCell,
        'nBRUCell': nBRUCell,
        'AdamW': AdamW,
        'SGDW': SGDW,
        'swish': swish
    }
)