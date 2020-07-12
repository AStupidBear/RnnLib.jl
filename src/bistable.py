import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import RNN, GRUCell
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import _caching_device
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import tf_utils


class BRUCell(DropoutRNNCellMixin, Layer):
    def __init__(self, units,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        self.units = units
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = self.units
        self.output_size = self.units
        super(BRUCell, self).__init__(**kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.kernelz = self.add_weight(
            name="kz",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.kernelr = self.add_weight(
            name="kr",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.kernelh = self.add_weight(
            name="kh",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.memoryz = self.add_weight(name="mz",
                                       shape=(self.units,),
                                       initializer=tf.keras.initializers.constant(
                                           1.0),
                                       caching_device=default_caching_device)
        self.memoryr = self.add_weight(
            name="mr",
            shape=(self.units,),
            initializer=tf.keras.initializers.constant(1.0),
            caching_device=default_caching_device)
        self.br = self.add_weight(
            name="br",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        self.bz = self.add_weight(
            name="bz",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        self.bh = self.add_weight(
            name="bh",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        super(BRUCell, self).build(input_shape)

    def call(self, inputs, states, training=None):
        prev_output = states[0] if nest.is_sequence(states) else states
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training, count=3)

        if 0. < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs

        if 0. < self.recurrent_dropout < 1.:
            prev_output_z = prev_output * rec_dp_mask[0]
            prev_output_r = prev_output * rec_dp_mask[1]
            prev_output_h = prev_output * rec_dp_mask[2]
        else:
            prev_output_z = prev_output
            prev_output_r = prev_output
            prev_output_h = prev_output

        r = tf.nn.tanh(tf.matmul(inputs_r, self.kernelr) +
                       prev_output_r * self.memoryr + self.br) + 1
        z = tf.nn.sigmoid(tf.matmul(inputs_z, self.kernelz) +
                          prev_output_z * self.memoryz + self.bz)
        h = tf.nn.tanh(tf.matmul(inputs_h, self.kernelh) +
                       r * prev_output_h + self.bh)
        output = (1.0 - z) * h + z * prev_output_h
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {
            'units':
                self.units,
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(BRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class nBRUCell(DropoutRNNCellMixin, Layer):
    def __init__(self, units,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        self.units = units
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = units
        self.output_size = self.units
        super(nBRUCell, self).__init__(**kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.kernelz = self.add_weight(
            name="kz",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.kernelr = self.add_weight(
            name="kr",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.kernelh = self.add_weight(
            name="kh",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.memoryz = self.add_weight(
            name="mz",
            shape=(self.units, self.units),
            initializer='orthogonal',
            caching_device=default_caching_device)
        self.memoryr = self.add_weight(
            name="mr",
            shape=(self.units, self.units),
            initializer='orthogonal',
            caching_device=default_caching_device)
        self.br = self.add_weight(
            name="br",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        self.bz = self.add_weight(
            name="bz",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        self.bh = self.add_weight(
            name="bh",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        super(nBRUCell, self).build(input_shape)

    def call(self, inputs, states, training=None):
        prev_output = states[0] if nest.is_sequence(states) else states
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training, count=3)

        if 0. < self.dropout < 1.:
            inputs_z = inputs * dp_mask[0]
            inputs_r = inputs * dp_mask[1]
            inputs_h = inputs * dp_mask[2]
        else:
            inputs_z = inputs
            inputs_r = inputs
            inputs_h = inputs

        if 0. < self.recurrent_dropout < 1.:
            prev_output_z = prev_output * rec_dp_mask[0]
            prev_output_r = prev_output * rec_dp_mask[1]
            prev_output_h = prev_output * rec_dp_mask[2]
        else:
            prev_output_z = prev_output
            prev_output_r = prev_output
            prev_output_h = prev_output

        r = tf.nn.tanh(tf.matmul(inputs_r, self.kernelr) +
                       tf.matmul(prev_output_r, self.memoryr + self.br)) + 1
        z = tf.nn.sigmoid(tf.matmul(inputs_z, self.kernelz) +
                          tf.matmul(prev_output_z, self.memoryz) + self.bz)
        h = tf.nn.tanh(tf.matmul(inputs_h, self.kernelh) +
                       r * prev_output_h + self.bh)
        output = (1.0 - z) * h + z * prev_output_h
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {
            'units':
                self.units,
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        base_config = super(nBRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def BRU(hidden_size,
        return_sequences=False,
        **kwargs):
    return RNN(BRUCell(hidden_size, **kwargs), return_sequences)


def nBRU(hidden_size,
         return_sequences=False,
         **kwargs):
    return RNN(nBRUCell(hidden_size, **kwargs), return_sequences)
