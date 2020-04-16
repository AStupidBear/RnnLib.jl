#!/usr/bin/env python

###################################################################################################
# args parsing

if True:
    import argparse
parser = argparse.ArgumentParser(description='rnnlib')
parser.add_argument('--model_path', type=str, default='model.h5')
parser.add_argument('--data_path', type=str, default='train.rnn')
parser.add_argument('--ckpt_fmt', default='ckpt-{epoch}.h5')
parser.add_argument('--log_dir', default='logs')
parser.add_argument('--feature_name', default='x')
parser.add_argument('--label_name', default='y')
parser.add_argument('--weight_name', default='w')
parser.add_argument('--pred_name', default='p')
parser.add_argument('--warm_start', type=int, default=0)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--sequence_size', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--layer', type=str, default='AHLN')
parser.add_argument('--out_activation', type=str, default='linear')
parser.add_argument('--hidden_sizes', type=str, default='128')
parser.add_argument('--loss', type=str, default='mse')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--kernel_sizes', type=str, default='7,9,11')
parser.add_argument('--pool_size', type=int, default=1)
parser.add_argument('--max_dilation', type=int, default=64)
parser.add_argument('--l2', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--use_batch_norm', type=int, default=0)
parser.add_argument('--use_skip_conn', type=int, default=0)
parser.add_argument('--bottleneck_size', type=int, default=32)
parser.add_argument('--out_dim', type=int, default=0)
parser.add_argument('--validation_split', type=float, default=0.2)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--warmup_epochs', type=int, default=3)
parser.add_argument('--factor', type=float, default=1)
parser.add_argument('--commission', type=float, default=1e-4)
parser.add_argument('--pnl_scale', type=float, default=1)
parser.add_argument('--close_thresh', type=float, default=0.5)
parser.add_argument('--eta', type=float, default=0.1)
args = parser.parse_args()
model_path, data_path, ckpt_fmt, log_dir = args.model_path, args.data_path, args.ckpt_fmt, args.log_dir
feature_name, label_name, weight_name = args.feature_name, args.label_name, args.weight_name
pred_name, warm_start, test, optimizer = args.pred_name, args.warm_start, args.test, args.optimizer
lr, batch_size, sequence_size, epochs = args.lr, args.batch_size, args.sequence_size, args.epochs
layer, out_activation, loss, kernel_size = args.layer, args.out_activation, args.loss, args.kernel_size
pool_size, max_dilation, dropout, l2 = args.pool_size, args.max_dilation, args.dropout, args.l2
use_batch_norm, use_skip_conn, bottleneck_size = args.use_batch_norm, args.use_skip_conn, args.bottleneck_size
out_dim, validation_split, patience = args.out_dim, args.validation_split, args.patience
warmup_epochs, factor, commission = args.warmup_epochs, args.factor, args.commission
pnl_scale, close_thresh, eta = args.pnl_scale, args.close_thresh, args.eta
hidden_sizes = list(map(int, args.hidden_sizes.split(',')))
kernel_sizes = list(map(int, args.kernel_sizes.split(',')))

loss = 'binary_crossentropy' if loss == 'bce' else loss
loss = 'categorical_crossentropy' if loss == 'cce' else loss
loss = 'sparse_categorical_crossentropy' if loss == 'spcce' else loss
out_activation = 'sigmoid' if loss == 'binary_crossentropy' else out_activation
out_activation = 'softmax' if 'categorical_crossentropy' in loss else out_activation
out_activation = 'tanh' if loss == 'pnl' else out_activation

###################################################################################################
# envs setting

if True:
    import os
if layer in ('LSTM', 'GRU'):
    os.environ['TF_DISABLE_MKL'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
usegpu = os.getenv('USE_GPU', '1') == '1'
if not usegpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

###################################################################################################
# package loading

import copy
import gc
import math
import re
import sys

import h5py
import hdf5plugin
import numpy as np
import tcn
import tensorflow as tf
import tensorflow.keras.backend as K
from alt_model_checkpoint.tensorflow import AltModelCheckpoint
from keras_adamw.optimizers_v2 import SGDW, AdamW
from keras_adamw.utils import get_weight_decays
from numba import jit
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ProgbarLogger, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.layers import (GRU, LSTM, Activation, AveragePooling1D,
                                     BatchNormalization, Conv1D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling1D,
                                     GlobalMaxPooling1D, Lambda, Layer,
                                     MaxPooling1D, SpatialDropout1D,
                                     TimeDistributed, add, concatenate)
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import HDF5Matrix, Sequence, multi_gpu_model
from tensorflow.python.client import device_lib

from ind_rnn import IndRNN
from lr_finder import LRFinder

###################################################################################################
# configuration

if os.getenv('TF_EAGER', '0') == '0':
    tf.compat.v1.disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
custom_objects = {'TCN': tcn.TCN, 'IndRNN': IndRNN, 'AdamW': AdamW, 'SGDW': SGDW}
print('current path %s\n' % os.getcwd())

###################################################################################################
# utility functions

def is_volta():
    for dev in device_lib.list_local_devices():
        m = re.findall('capability: ([0-9\.]+)', dev.physical_device_desc)
        if len(m) > 0:
            capability = float(m[0])
            if capability > 7:
                return True
    return False

def use_mixed_precision():
    return is_volta()

###################################################################################################
# custom architectures


def OnnxConv(*args, **kwargs):
    def conv(o):
        for i in range(10):
            o = Conv1D(10, 3, padding='causal')(o)
            o = Activation('relu')(o)
        return o
    return conv


def ResNet(filters,
           kernel_size,
           pool_size,
           padding='causal',
           dropout=0.0,
           return_sequences=False,
           use_batch_norm=True):
    def resnet_module(i, factor):
        o = Conv1D(filters, factor * kernel_size - 1, padding=padding)(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        return o

    def resnet(i):
        o = resnet_module(i, 3)
        o = resnet_module(o, 2)
        o = resnet_module(o, 1)
        o = Conv1D(filters, kernel_size, padding=padding)(o)
        if use_batch_norm:
            o = BatchNormalization()(o)
        # expand channels for the sum
        if filters != i.shape[-1]:
            i = Conv1D(filters, 1, padding=padding)(i)
        if use_batch_norm:
            i = BatchNormalization()(i)
        o = add([i, o])
        o = Activation('relu')(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        if return_sequences:
            o = CausalAveragePooling1D(pool_size)(o)
        else:
            o = GlobalAveragePooling1D()(o)
        return o
    return resnet


def MLP(hidden_size,
        dropout=0.0,
        use_batch_norm=True):
    def mlp(i):
        o = Dense(hidden_size)(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        if dropout > 0:
            o = Dropout(dropout)(o)
        return o
    return mlp


def Conv(filters,
         dropout=0.0,
         use_batch_norm=True,
         return_sequences=False):
    def conv(i):
        o = Conv1D(filters, 1, padding='causal')(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        if not return_sequences:
            o = Lambda(lambda x: x[:, -1, :])(o)
        return o
    return conv


def Rocket(filters,
           pool_size,
           max_dilation,
           kernel_sizes=(7, 9, 11),
           padding='causal',
           return_sequences=False):
    def rocket(i):
        outs = []
        dilations = [2**n for n in range(10) if 2**n <= max_dilation]
        filters_ = int(100 * filters / len(dilations) / len(kernel_sizes))
        if filters_ > 1:
            for kernel_size in kernel_sizes:
                for dilation in dilations:
                    o = Conv1D(filters_, kernel_size, padding=padding, dilation_rate=dilation, trainable=False,
                               kernel_initializer=RandomNormal(stddev=1), bias_initializer=RandomUniform(-1, 1))(i)
                    outs.append(o)
        else:
            for n in range(100 * filters):
                kernel_size = int(np.random.choice(kernel_sizes))
                dilation = int(np.random.choice(dilations))
                o = Conv1D(1, kernel_size, padding=padding, dilation_rate=dilation, trainable=False,
                           kernel_initializer=RandomNormal(stddev=1), bias_initializer=RandomUniform(-1, 1))(i)
                outs.append(o)
        o = concatenate(outs)
        if return_sequences:
            o_max = CausalMaxPooling1D(pool_size)(o)
            o_avg = CausalAveragePooling1D(pool_size)(o)
        else:
            o_max = GlobalMaxPooling1D()(o)
            o_avg = GlobalAveragePooling1D()(o)
        o = concatenate([o_max, o_avg])
        return o
    return rocket


def Inception(filters,
              kernel_size,
              pool_size,
              padding='causal',
              dropout=0.0,
              return_sequences=False,
              use_batch_norm=True,
              bottleneck_size=32):
    def inception_module(i):
        kernel_sizes = [kernel_size * (2 ** i) for i in range(3)]
        conv_list = [Conv1D(filters // 4, 1, padding=padding,
                            use_bias=False)(CausalMaxPooling1D(3)(i))]
        if bottleneck_size > 0 and int(i.shape[-1]) > 4 * bottleneck_size:
            i = Conv1D(bottleneck_size, 1, padding=padding, use_bias=False)(i)
        for ks in kernel_sizes:
            o = Conv1D(filters // 4, ks, padding=padding, use_bias=False)(i)
            conv_list.append(o)
        o = concatenate(conv_list)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        return o

    def inception(i):
        o = inception_module(i)
        o = inception_module(o)
        o = inception_module(o)
        i = Conv1D(int(o.shape[-1]), 1, padding=padding, use_bias=False)(i)
        if use_batch_norm:
            i = BatchNormalization()(i)
        o = add([i, o])
        o = Activation('relu')(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        if return_sequences:
            o = CausalAveragePooling1D(pool_size)(o)
        else:
            o = GlobalAveragePooling1D()(o)
        return o
    return inception


def TCN(filters,
        kernel_size,
        pool_size,
        max_dilation,
        padding='causal',
        dropout=0.0,
        return_sequences=False,
        activation='relu',
        use_batch_norm=True):
    def _tcn(i):
        dilations = [2**n for n in range(10) if 2**n <= max_dilation]
        o = tcn.TCN(filters, kernel_size, 1, dilations=dilations, padding=padding, dropout_rate=dropout,
                    use_batch_norm=use_batch_norm, activation=activation, return_sequences=True)(i)
        if return_sequences:
            o = CausalAveragePooling1D(pool_size)(o)
        else:
            o = GlobalAveragePooling1D()(o)
        return o
    return _tcn


def ResRNN(hidden_size,
           dropout=0.0,
           return_sequences=True,
           use_skip_conn=False,
           layer='LSTM'):
    def rnn(i):
        o = eval(layer)(hidden_size, dropout=dropout,
                        return_sequences=return_sequences)(i)
        o = eval(layer)(hidden_size, dropout=dropout,
                        return_sequences=return_sequences)(i)
        if hidden_size != i.shape[-1]:
            i = DenseMod(hidden_size)(i)
        if use_skip_conn:
            o = add([i, o])
        return o
    return rnn


def AHLN(hidden_size,
         max_pool_size,
         dropout=0.0,
         return_sequences=True,
         use_batch_norm=True,
         use_skip_conn=False):
    def ahln(i):
        pool_list = [i]
        for pool_size in [4**n for n in range(10) if 4**n <= max_pool_size]:
            pool_list.append(CausalAveragePooling1D(pool_size)(i))
            pool_list.append(CausalMaxPooling1D(pool_size)(i))
            pool_list.append(CausalMinPooling1D(pool_size)(i))
        o = concatenate(pool_list)
        for n in range(2):
            o = DenseMod(hidden_size)(o)
            if use_batch_norm:
                o = BatchNormalization()(o)
            o = Activation('relu')(o)
            if dropout > 0:
                o = SpatialDropout1D(dropout)(o)
        if hidden_size != i.shape[-1]:
            i = DenseMod(hidden_size)(i)
        if use_batch_norm:
            i = BatchNormalization()(i)
        if use_skip_conn:
            o = add([i, o])
        o = Activation('relu')(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        if not return_sequences:
            o = GlobalAveragePooling1D()(o)
        return o
    return ahln


def CausalAveragePooling1D(pool_size):
    def pool(i):
        if pool_size > 1:
            o = Lambda(lambda x: K.temporal_padding(x, (pool_size - 1, 0)))(i)
            o = AveragePooling1D(pool_size, strides=1, padding='valid')(o)
        else:
            o = i
        return o
    return pool


def CausalMaxPooling1D(pool_size):
    def pool(i):
        if pool_size > 1:
            o = Lambda(lambda x: K.temporal_padding(x, (pool_size - 1, 0)))(i)
            o = MaxPooling1D(pool_size, strides=1, padding='valid')(o)
        else:
            o = i
        return o
    return pool


def CausalMinPooling1D(pool_size):
    def pool(i):
        if pool_size > 1:
            o = Lambda(lambda x: -K.temporal_padding(x, (pool_size - 1, 0)))(i)
            o = MaxPooling1D(pool_size, strides=1, padding='valid')(o)
            o = Lambda(lambda x: -x)(o)
        else:
            o = i
        return o
    return pool


def DenseMod(units, activation=None):
    def densemod(i):
        if 'ngraph_bridge' in sys.modules and len(i.shape) == 3:
            return Conv1D(units, 1, padding='causal', activation=activation)(i)
        else:
            return Dense(units, activation=activation)(i)
    return densemod


###################################################################################################
# custom loss functions

def pnl(y_true, y_pred, c=commission, λ=pnl_scale):
    r, p, c = λ * y_true, y_pred, λ * c
    l = - K.mean(r * p)
    if len(y_pred.get_shape()) < 3:
        return l
    c1 = c / K.cast(K.shape(p)[1], 'float32')
    if c > 0:
        if T > 1:
            Δp = p[:, 1:, :] - p[:, :-1, :]
            l += c * K.mean(K.abs(Δp))
        l += c1 * K.mean(K.abs(p[:, 0, :]))
        l += c1 * K.mean(K.abs(p[:, -1, :]))
    return l


@jit(nopython=True)
def loss_augmented_inference(r, z, λ=pnl_scale, c=commission, ϵ=eta, η=close_thresh):
    N, T = r.shape[0], r.shape[1]
    Q = np.zeros((N, T, 3, 5), np.float32)
    π = np.zeros((N, T, 1), np.int8)
    M = np.array([[-1, -1, -1, 0, 1], [-1, 0, 0, 0, 1],
                  [-1, 0, 1, 1, 1]], np.int8)
    # M = np.array([[-1, -1, 0, 0, 1], [-1, 0, 0, 0, 1], [-1, 0, 0, 1, 1]], dtype = 'int8')
    for n in range(N):
        Vᵗ = np.zeros(3, np.float32)
        Ṽᵗ = np.zeros(3, np.float32)
        for t in range(T - 1, -1, -1):
            rₙₜ, zₙₜ = r[n, t, 0], z[n, t, 0]
            for s in [-1, 0, 1]:
                for a in range(5):
                    s̃ = 0 if t == T else M[s + 1, a]
                    a_, b_ = np.sign(a - 2), abs(a - 2) / 2
                    c_ = a_ * (b_ * zₙₜ + 1 - b_)
                    Q[n, t, s + 1, a] = max(c_, (η + 1) / 2)
                    Q[n, t, s + 1, a] += λ * ϵ * \
                        (Ṽᵗ[s̃ + 1] + rₙₜ * s̃ - c * abs(s̃ - s))
                    Vᵗ[s + 1] = Q[n, t, s + 1, :].max()
            for i in range(3):
                Ṽᵗ[i] = Vᵗ[i]
    for n in range(N):
        s = 0
        for t in range(T):
            π[n, t, 0] = Q[n, t, s + 1, :].argmax()
            s = M[s + 1, π[n, t, 0]]
    return π


def direct(y_true, y_pred, η=close_thresh):
    def score(z, y):
        a = K.sign(y - 2)
        b = K.abs(y - 2) / 2
        c = a * (b * z + 1 - b)
        return K.maximum(c, (η + 1) / 2)
    s1 = score(y_pred[:, :, 1], y_true[:, :, 1])
    s2 = score(y_pred[:, :, 2], y_true[:, :, 2])
    return s1 - s2


@jit(nopython=True)
def direct_loss(r, z, λ=pnl_scale, c=commission, η=close_thresh):
    N, T = r.shape[0], r.shape[1]
    M = np.array([[-1, -1, -1, 0, 1], [-1, 0, 0, 0, 1],
                  [-1, 0, 1, 1, 1]], np.int8)
    l = 0.0
    for n in range(N):
        s = 0
        for t in range(T):
            a_, b_ = np.sign(z[n, t, 0]), np.abs(z[n, t, 0])
            c_ = 2 if b_ > 1 else (1 if b_ > η else 0)
            s̃ = 0 if t == T else M[s + 1, int(a_ * c_ + 2)]
            l += - r[n, t, 0] * s̃ + c * np.abs(s̃ - s)
            s = s̃
    return λ * l / N / T


###################################################################################################
# custom data loader

class JLSequence(Sequence):

    def __init__(self, data_path, sequence_size, batch_size, logger):
        self.sess = tf.compat.v1.keras.backend.get_session()
        if not os.path.isfile(data_path):
            F, N, T = 30, 3000, 4802
            with h5py.File(data_path, 'w') as fid:
                x = np.random.randn(T, N, F).astype('float32')
                y = np.mean(x, axis=2)
                fid.create_dataset(feature_name, data=x, chunks=(T // 2, 2 * batch_size, F), **hdf5plugin.Blosc('zstd'))
                fid.create_dataset(label_name, data=y, chunks=(T // 2, 2 * batch_size), **hdf5plugin.Blosc('zstd'))
        self.fid = h5py.File(data_path, 'r+', rdcc_nbytes=1024**3, rdcc_nslots=100000)
        if label_name in self.fid.keys():
            self.y = self.fid[label_name][()]
        else:
            self.y = None
        if pred_name in self.fid.keys():
            self.p = self.fid[pred_name]
        else:
            self.p = None
        if weight_name in self.fid.keys():
            self.w = self.fid[weight_name]
        else:
            self.w = None
        import time;
        ti = time.time()
        self.x = self.fid[feature_name]
        if sequence_size == 0:
            sequence_size = self.x.shape[0]
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.n_sequences = math.floor(self.x.shape[0] / sequence_size)
        self.n_batches = math.ceil(self.x.shape[1] / batch_size)
        self.start = 0
        self.end = self.n_sequences * self.n_batches
        self.logger = logger
        self.out_seq = self.y.shape[0] == self.x.shape[0]
        if out_dim < 1:
            if self.out_seq:
                self.out_dim = self.y.shape[-1] if self.y.ndim == 3 else 1
            else:
                self.out_dim = self.y.shape[-1] if self.y.ndim == 2 else 1
        else:
            self.out_dim = out_dim

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        idx = idx + self.start
        n, t = idx % self.n_batches, idx // self.n_batches
        ts = slice(self.sequence_size * t, self.sequence_size * (t + 1))
        ns = slice(self.batch_size * n, self.batch_size * (n + 1))
        ns = slice(ns.start, min(ns.stop, self.x.shape[1]))
        x = self.x[ts, ns, :].swapaxes(0, 1)
        if x.dtype == 'uint8':
            x = x / 128 - 1
        if self.y is not None:
            if self.y.shape[0] == self.x.shape[0]:
                y = self.y[ts, ns].swapaxes(0, 1)
                y = y.reshape(*y.shape[:2], -1)
            else:
                y = self.y[ns]
                y = y.reshape(*y.shape[:1], -1)
        else:
            y = None
        if self.w is not None:
            if self.w.shape[0] == self.x.shape[0]:
                w = self.w[ts, ns].swapaxes(0, 1)
            else:
                w = self.w[ns]
        else:
            w = None
        if w is not None and loss in  ('pnl', 'direct'):
            w = w.reshape(*w.shape, 1)
            y = np.multiply(y, w)
            w = None
        if loss == 'direct' and y is not None:
            if tf.executing_eagerly():
                z = model(x).numpy()
            else:
                tf.compat.v1.keras.backend.set_session(self.sess)
                with self.sess.graph.as_default():
                    z = model.predict_on_batch(x)
            z = z.astype('float', copy=False)
            self.logger.add_log('direct_loss', direct_loss(y, z))
            yw = loss_augmented_inference(y, z, 0)
            yϵ = loss_augmented_inference(y, z)
            y = np.concatenate((y, yw, yϵ), axis=-1)
            return x, y, [w]
        else:
            return x, y, [w]

    def split(self, split_at):
        if split_at is None:
            return self, None
        else:
            trn_gen, val_gen = copy.copy(gen), copy.copy(gen)
            val_gen.start = trn_gen.end = math.floor(len(gen) * (1 - split_at))
            return trn_gen, val_gen

    def fill_pred(self, pred):
        npred = 0
        for idx in range(len(self)):
            idx = idx + self.start
            n, t = idx % self.n_batches, idx // self.n_batches
            ts = slice(self.sequence_size * t, self.sequence_size * (t + 1))
            ns = slice(self.batch_size * n, self.batch_size * (n + 1))
            ns = range(ns.start, min(ns.stop, self.x.shape[1]))
            y = pred[npred:(npred + len(ns)), :, 0:self.p.shape[2]]
            if not self.p:
                shape = (*self.x.shape[:2], y.shape[-1])
                self.fid.create_dataset(pred_name, shape, dtype='float')
            self.p[ts, ns, :] = y.swapaxes(0, 1)
            self.fid.flush()
            npred += len(ns)


class CustomProgbarLogger(ProgbarLogger):

    def __init__(self):
        super(CustomProgbarLogger, self).__init__()
        self.use_steps = True
        self.logs = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs.update(self.logs)
        super(CustomProgbarLogger, self).on_batch_end(batch, logs)

    def add_log(self, k, v):
        self.logs[k] = v
        if hasattr(self, 'params'):
            self.params['metrics'].append(k)


###################################################################################################
# configuration

# horovod init
try:
    import horovod.tensorflow.keras as hvd
    hvd.init()
    world_size = hvd.size()
    use_horovod = world_size > 1
    local_rank = hvd.local_rank()
except:
    world_size = 1
    use_horovod = False
    local_rank = 0

# set gpu specific options
gpus = tf.config.list_physical_devices('GPU')
if test == 0 and usegpu and gpus:
    tf.config.experimental.set_visible_devices(
        gpus[local_rank % len(gpus)], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # mixed precision
    if use_mixed_precision():
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
elif os.getenv('USE_NGRAPH', '0') == '1':
    import ngraph_bridge

# load hdf5 data or generate dummy data
logger = CustomProgbarLogger()
gen = JLSequence(data_path, sequence_size, batch_size, logger)
trn_gen, val_gen = gen.split(validation_split)

###################################################################################################
# model testing

if test == 1:
    model = load_model(model_path, custom_objects=custom_objects)
    gen.fill_pred(model.predict(gen))
    exit()

###################################################################################################
# model building

# expected input batch shape: (N, T, F)
T, N, F = gen.x.shape
pool_size = min(T // 3, pool_size)
max_dilation = min(T // kernel_size, max_dilation)
if layer == 'MLP':
    o = i = Input(shape=(T, F))
    o = Flatten()(o)
elif layer in ('IndRNN', 'PLSTM'):
    o = i = Input(shape=(T, F))
elif os.getenv('USE_TFLITE', '0') == '1':
    o = i = Input(shape=(T, F), batch_size=batch_size)
else:
    o = i = Input(shape=(None, F))
for (l, h) in enumerate(hidden_sizes):
    return_sequences = l + 1 < len(hidden_sizes) or gen.out_seq
    if layer == 'MLP':
        o = MLP(h, dropout=dropout, use_batch_norm=use_batch_norm)(o)
    elif layer == 'Conv':
        o = Conv(h, dropout=dropout, use_batch_norm=use_batch_norm,
                 return_sequences=return_sequences)(o)
    elif layer == 'ResNet':
        o = ResNet(h, kernel_size, pool_size, dropout=dropout,
                   use_batch_norm=use_batch_norm, return_sequences=return_sequences)(o)
    elif layer == 'Inception':
        o = Inception(h, kernel_size, pool_size, dropout=dropout, use_batch_norm=use_batch_norm,
                      return_sequences=return_sequences, bottleneck_size=bottleneck_size)(o)
    elif layer == 'Rocket':
        if l == 0:
            o = Rocket(h, pool_size, max_dilation, kernel_sizes,
                       return_sequences=return_sequences)(o)
        else:
            o = DenseMod(h, activation='relu')(o)
    elif layer == 'TCN':
        o = TCN(h, kernel_size, pool_size, max_dilation, dropout=dropout,
                use_batch_norm=use_batch_norm, return_sequences=return_sequences)(o)
    elif layer == 'AHLN':
        o = AHLN(h, max_dilation, dropout=dropout, use_batch_norm=use_batch_norm,
                 use_skip_conn=use_skip_conn, return_sequences=return_sequences)(o)
    else:
        o = ResRNN(h, dropout=dropout, return_sequences=return_sequences,
                   use_skip_conn=use_skip_conn, layer=layer)(o)
o = Activation(out_activation, dtype='float')(DenseMod(gen.out_dim)(o))
if loss == 'direct':
    o = concatenate([o, o, o])
model = Model(inputs=[i], outputs=[o])
# print(model.summary())

# warm start
resume_from_epoch = 0
if warm_start == 1:
    for try_epoch in range(epochs, 0, -1):
        h5 = log_dir + '/' + ckpt_fmt.format(epoch=try_epoch)
        if os.path.isfile(h5):
            resume_from_epoch = try_epoch
            model = load_model(h5, custom_objects=custom_objects)
            break
else:
    import shutil
    shutil.rmtree(log_dir, ignore_errors=True)

# multi-gpu
base_model = model
if not use_horovod and len(gpus) > 1:
    model = multi_gpu_model(model, len(gpus), cpu_relocation=True)

# callbacks
callbacks = []
if use_horovod:
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(
        warmup_epochs, verbose=1))
if loss == 'direct':
    callbacks.append(logger)
if local_rank == 0:
    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    callbacks.append(AltModelCheckpoint(log_dir + '/' + ckpt_fmt, base_model))
    callbacks.append(TensorBoard(log_dir))
    if validation_split > 0:
        callbacks.append(EarlyStopping(patience=patience, verbose=1))
    if factor < 1:
        callbacks.append(ReduceLROnPlateau(
            'loss', factor, patience, min_lr=1e-6, varbose=1))

# optimizer
weight_decays = {l: l2 for l in get_weight_decays(model)}
if not base_model.optimizer:
    # Horovod: adjust learning rate based on number of GPUs.
    if optimizer == 'SGDW':
        opt = SGDW(10 * lr * world_size, momentum = 0.9, nesterov=True, weight_decays=weight_decays)
    elif optimizer == 'AdamW':
        opt = AdamW(lr * world_size, weight_decays=weight_decays)
    else:
        opt = eval(optimizer)(lr * world_size)
    if not use_mixed_precision():
        opt.clipnorm = 1
else:
    # use serial model's optimizer state
    opt = base_model.optimizer
opt = hvd.DistributedOptimizer(opt) if use_horovod else opt

# compile
sample_weight_mode = None if gen.w is None or not gen.out_seq else 'temporal'
if 'mse' in str(loss):
    metric = ['mae']
elif 'crossentropy' in str(loss):
    metric = ['acc']
    if 'binary' in str(loss):
        metric.append(pnl)
else:
    metric = None
lossf = eval(loss) if loss in ('pnl', 'direct') else loss
model.compile(loss=lossf, optimizer=opt, metrics=metric,
              sample_weight_mode=sample_weight_mode,
              experimental_run_tf_function=False)

###################################################################################################
# model building

# lr finder
if lr == 0 or layer == 'Rocket' and lr >= 1e-3:
    lr_finder = LRFinder(model)
    epochs_ = max(1, 100 * batch_size // N)
    lr_finder.find_generator(gen, 1e-6, 1e-2, epochs_)
    best_lr = min(1e-3, lr_finder.get_best_lr(5))
    K.set_value(model.optimizer.lr, best_lr)
    print(40 * '=', '\nSet lr to: %s\n' % best_lr,  40 * '=')

# train model
model.fit(
    x=trn_gen,
    epochs=epochs,
    verbose=1 if local_rank == 0 else 0,
    callbacks=callbacks,
    validation_data=val_gen,
    shuffle=True,
    initial_epoch=resume_from_epoch,
    steps_per_epoch=len(gen) // world_size,
    validation_steps=len(val_gen) // world_size,
    workers=0 if loss == 'direct' else 4,
)
base_model.save(model_path)
