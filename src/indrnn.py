from __future__ import absolute_import
import warnings

from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, RNN, InputSpec, LayerNormalization
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin, _caching_device, _generate_zero_filled_state_for_cell
# from tensorflow.python.keras.layers.recurrent import _config_for_enable_caching_device
from tensorflow.python.util import nest
from tensorflow.python.keras.utils import tf_utils


class IndRNNCell(DropoutRNNCellMixin, Layer):
    def __init__(self, units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer=None,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 use_batch_norm=False,
                 **kwargs):
        super(IndRNNCell, self).__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer) \
            if recurrent_initializer is not None else None
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.use_batch_norm = use_batch_norm
        self.state_size = self.units
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            name='input_kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        if self.recurrent_initializer is None:
            self.recurrent_initializer = initializers.RandomUniform(-1.0, 1.0)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units,),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = K.clip(self.recurrent_kernel, -1, 1)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
        else:
            self.bias = None
        if self.use_batch_norm:
            self.norm = LayerNormalization()
        else:
            self.norm = lambda x: x
        self.built = True

    def call(self, inputs, states, training=None):
        prev_output = states[0] if nest.is_sequence(states) else states
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, self.kernel)
        else:
            h = K.dot(inputs, self.kernel)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask
        output = self.norm(h) + prev_output * self.recurrent_kernel
        if self.activation is not None:
            output = self.activation(output)
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout
        }
        # config.update(_config_for_enable_caching_device(self))
        base_config = super(IndRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def IndRNN(hidden_size,
           return_sequences=False,
           **kwargs):
    return RNN(IndRNNCell(hidden_size, **kwargs), return_sequences)
