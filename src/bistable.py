import tensorflow as tf
from tensorflow.keras.layers import Layer, RNN


class BRUCell(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(BRUCell, self).__init__(units, **kwargs)

    def build(self, input_shape):
        self.kernelz = self.add_weight(name="kz", shape=(input_shape[1], self.units), dtype=tf.float32,
                                       initializer='glorot_uniform')
        self.kernelr = self.add_weight(name="kr", shape=(input_shape[1], self.units), dtype=tf.float32,
                                       initializer='glorot_uniform')
        self.kernelh = self.add_weight(name="kh", shape=(input_shape[1], self.units), dtype=tf.float32,
                                       initializer='glorot_uniform')

        self.memoryz = self.add_weight(name="mz", shape=(
            self.units,), dtype=tf.float32, initializer=tf.keras.initializers.constant(1.0))
        self.memoryr = self.add_weight(name="mr", shape=(
            self.units,), dtype=tf.float32, initializer=tf.keras.initializers.constant(1.0))

        self.br = self.add_weight(name="br", shape=(
            self.units,), dtype=tf.float32, initializer='zeros')
        self.bz = self.add_weight(name="bz", shape=(
            self.units,), dtype=tf.float32, initializer='zeros')

        super(BRUCell, self).build(input_shape)

    def call(self, input, states):
        prev_out = states[0]
        r = tf.nn.tanh(tf.matmul(input, self.kernelr) +
                       prev_out * self.memoryr + self.br) + 1
        z = tf.nn.sigmoid(tf.matmul(input, self.kernelz) +
                          prev_out * self.memoryz + self.bz)
        h = tf.nn.tanh(tf.matmul(input, self.kernelh) + r * prev_out)
        output = (1.0 - z) * h + z * prev_out
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        return [tf.zeros(shape=(batch_size, self.units), dtype=dtype)]

    def get_config(self):
        config = {'units': self.units}
        base_config = super(BRUCell, self).get_config()
        del base_config['trainable']
        return dict(list(base_config.items()) + list(config.items()))


class nBRUCell(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(nBRUCell, self).__init__(units, **kwargs)

    def build(self, input_shape):
        self.kernelz = self.add_weight(name="kz", shape=(input_shape[1], self.units), dtype=tf.float32,
                                       initializer='glorot_uniform')
        self.kernelr = self.add_weight(name="kr", shape=(input_shape[1], self.units), dtype=tf.float32,
                                       initializer='glorot_uniform')
        self.kernelh = self.add_weight(name="kh", shape=(input_shape[1], self.units), dtype=tf.float32,
                                       initializer='glorot_uniform')

        self.memoryz = self.add_weight(name="mz", shape=(self.units, self.units), dtype=tf.float32,
                                       initializer='orthogonal')
        self.memoryr = self.add_weight(name="mr", shape=(self.units, self.units), dtype=tf.float32,
                                       initializer='orthogonal')

        self.br = self.add_weight(name="br", shape=(
            self.units,), dtype=tf.float32, initializer='zeros')
        self.bz = self.add_weight(name="bz", shape=(
            self.units,), dtype=tf.float32, initializer='zeros')

        super(nBRUCell, self).build(input_shape)

    def call(self, input, states):
        prev_out = states[0]
        z = tf.nn.sigmoid(tf.matmul(input, self.kernelz) +
                          tf.matmul(prev_out, self.memoryz) + self.bz)
        r = tf.nn.tanh(tf.matmul(input, self.kernelr) +
                       tf.matmul(prev_out, self.memoryr) + self.br) + 1
        h = tf.nn.tanh(tf.matmul(input, self.kernelh) + r * prev_out)
        output = (1.0 - z) * h + z * prev_out
        return output, [output]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        return [tf.zeros(shape=(batch_size, self.units), dtype=dtype)]

    def get_config(self):
        config = {'units': self.units}
        base_config = super(nBRUCell, self).get_config()
        del base_config['trainable']
        return dict(list(base_config.items()) + list(config.items()))


def BRU(hidden_size,
        dropout=0.0,
        return_sequences=False):
    cell = BRUCell(hidden_size)
    return RNN(cell, return_sequences)


def nBRU(hidden_size,
         dropout=0.0,
         return_sequences=False):
    cell = nBRUCell(hidden_size)
    return RNN(cell, return_sequences)
