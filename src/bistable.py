import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, RNN, GRUCell, LayerNormalization
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, _caching_device, _generate_zero_filled_state_for_cell
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import tf_utils


class BRUCell(DropoutRNNCellMixin, Layer):
    def __init__(self, units,
                 dropout=0.,
                 recurrent_dropout=0.,
                 use_batch_norm=False,
                 **kwargs):
        self.units = units
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.use_batch_norm = use_batch_norm
        self.state_size = self.units
        self.output_size = self.units
        super(BRUCell, self).__init__(**kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.kernel_z = self.add_weight(
            name="kernel_z",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.kernel_r = self.add_weight(
            name="kernel_r",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.kernel_h = self.add_weight(
            name="kernel_h",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.memory_z = self.add_weight(
            name="memory_z",
            shape=(self.units,),
            initializer=tf.keras.initializers.constant(1.0),
            caching_device=default_caching_device)
        self.memory_r = self.add_weight(
            name="memory_r",
            shape=(self.units,),
            initializer=tf.keras.initializers.constant(1.0),
            caching_device=default_caching_device)
        self.bias_r = self.add_weight(
            name="bias_r",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        self.bias_z = self.add_weight(
            name="bias_z",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        self.bias_h = self.add_weight(
            name="bias_h",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        if self.use_batch_norm:
            self.norm_rx = LayerNormalization()
            self.norm_rh = LayerNormalization()
            self.norm_zx = LayerNormalization()
            self.norm_zh = LayerNormalization()
            self.norm_hx = LayerNormalization()
            self.norm_hh = LayerNormalization()
        else:
            self.norm_rx = self.norm_zx = self.norm_hx = lambda x: x
            self.norm_rh = self.norm_zh = self.norm_hh = lambda x: x
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

        r = K.tanh(self.norm_rx(K.dot(inputs_r, self.kernel_r)) +
                   self.norm_rh(prev_output_r * self.memory_r) + self.bias_r) + 1
        z = K.sigmoid(self.norm_zx(K.dot(inputs_z, self.kernel_z)) +
                      self.norm_zh(prev_output_z * self.memory_z) + self.bias_z)
        h = K.tanh(self.norm_hx(K.dot(inputs_h, self.kernel_h)) +
                   r * prev_output_h + self.bias_h)
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
                self.recurrent_dropout,
            'use_batch_norm':
                self.use_batch_norm
        }
        base_config = super(BRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class nBRUCell(DropoutRNNCellMixin, Layer):
    def __init__(self, units,
                 dropout=0.,
                 recurrent_dropout=0.,
                 use_batch_norm=False,
                 **kwargs):
        self.units = units
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.use_batch_norm = use_batch_norm
        self.state_size = units
        self.output_size = self.units
        super(nBRUCell, self).__init__(**kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.kernel_z = self.add_weight(
            name="kernel_z",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.kernel_r = self.add_weight(
            name="kernel_r",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.kernel_h = self.add_weight(
            name="kernel_h",
            shape=(input_shape[1], self.units),
            initializer='glorot_uniform',
            caching_device=default_caching_device)
        self.memory_z = self.add_weight(
            name="memory_z",
            shape=(self.units, self.units),
            initializer='orthogonal',
            caching_device=default_caching_device)
        self.memory_r = self.add_weight(
            name="memory_r",
            shape=(self.units, self.units),
            initializer='orthogonal',
            caching_device=default_caching_device)
        self.bias_r = self.add_weight(
            name="bias_r",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        self.bias_z = self.add_weight(
            name="bias_z",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        self.bias_h = self.add_weight(
            name="bias_h",
            shape=(self.units,),
            initializer='zeros',
            caching_device=default_caching_device)
        if self.use_batch_norm:
            self.norm_rx = LayerNormalization()
            self.norm_rh = LayerNormalization()
            self.norm_zx = LayerNormalization()
            self.norm_zh = LayerNormalization()
            self.norm_hx = LayerNormalization()
            self.norm_hh = LayerNormalization()
        else:
            self.norm_rx = self.norm_zx = self.norm_hx = lambda x: x
            self.norm_rh = self.norm_zh = self.norm_hh = lambda x: x
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

        r = K.tanh(self.norm_rx(K.dot(inputs_r, self.kernel_r)) +
                   self.norm_rh(K.dot(prev_output_r, self.memory_r)) + self.bias_r) + 1
        z = K.sigmoid(self.norm_zx(K.dot(inputs_z, self.kernel_z)) +
                      self.norm_zh(K.dot(prev_output_z, self.memory_z)) + self.bias_z)
        h = K.tanh(self.norm_hx(K.dot(inputs_h, self.kernel_h)) +
                   r * prev_output_h + self.bias_h)
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
                self.recurrent_dropout,
            'use_batch_norm':
                self.use_batch_norm
        }
        base_config = super(BRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def BRU(hidden_size,
        return_sequences=False,
        **kwargs):
    return RNN(BRUCell(hidden_size, **kwargs), return_sequences)


def nBRU(hidden_size,
         return_sequences=False,
         **kwargs):
    return RNN(nBRUCell(hidden_size, **kwargs), return_sequences)
