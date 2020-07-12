#!/usr/bin/env python

###################################################################################################
# args parsing
if True:
    import argparse
parser = argparse.ArgumentParser(description='rnnlib')

parser.add_argument('--layer', type=str, default='AHLN')
parser.add_argument('--hidden_sizes', type=str, default='128')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--kernel_sizes', type=str, default='7,9,11')
parser.add_argument('--recept_field', type=int, default=64)
parser.add_argument('--pool_size', type=int, default=1)
parser.add_argument('--bottleneck_size', type=int, default=32)
parser.add_argument('--use_skip_conn', action='store_true')
parser.add_argument('--input_dim', type=int, default=5000)
parser.add_argument('--embed_dim', type=int, default=32)
parser.add_argument('--output_dim', type=int, default=0)
parser.add_argument('--out_activation', type=str, default='linear')

parser.add_argument('--l2', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--use_batch_norm', action='store_true')

parser.add_argument('--loss', type=str, default='mse') 
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--patience', type=int, default=1000)
parser.add_argument('--warmup_epochs', type=int, default=3)

parser.add_argument('--warm_start', action='store_true')
parser.add_argument('--reset_epoch', action='store_true')
parser.add_argument('--test', action='store_true')
parser.add_argument('--prefetch', action='store_true')
parser.add_argument('--sequence_size', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--validation_split', type=float, default=0.0)
parser.add_argument('--use_multiprocessing', action='store_true')

parser.add_argument('--model_path', type=str, default='model.h5')
parser.add_argument('--data_path', type=str, default='train.rnn')
parser.add_argument('--pred_path', type=str, default='pred.rnn')

parser.add_argument('--ckpt_fmt', default='ckpt-{epoch}.h5')
parser.add_argument('--log_dir', default='logs')

parser.add_argument('--feature_name', default='x')
parser.add_argument('--label_name', default='y')
parser.add_argument('--weight_name', default='w')
parser.add_argument('--pred_name', default='p')

parser.add_argument('--debug', action='store_true')
parser.add_argument('--eager', action='store_true')

parser.add_argument('--factor', type=float, default=1)
parser.add_argument('--commission', type=float, default=1e-4)
parser.add_argument('--pnl_scale', type=float, default=1)
parser.add_argument('--close_thresh', type=float, default=0.5)
parser.add_argument('--eta', type=float, default=0.1)

args = parser.parse_args()

###################################################################################################
# envs setting

if True:
    import os
if args.layer in ('LSTM', 'GRU'):
    os.environ['TF_DISABLE_MKL'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
usegpu = os.getenv('USE_GPU', '1') == '1'
if not usegpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import logging
logging.getLogger('tensorflow').disabled = True

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
import tensorflow as tf
import tensorflow.keras.backend as K
from alt_model_checkpoint.tensorflow import AltModelCheckpoint
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
                                     GlobalMaxPooling1D, Lambda, Layer, RNN,
                                     MaxPooling1D, SpatialDropout1D,
                                     TimeDistributed, Add, Concatenate, Embedding)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import HDF5Matrix, Sequence, multi_gpu_model
from tensorflow.python.client import device_lib

from lrfinder import LRFinder
from custom import *

###################################################################################################
# configuration

if not args.eager and not args.debug:
    tf.compat.v1.disable_eager_execution()
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

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
           use_batch_norm=False):
    def resnet_module(i, factor, activation=None, dropout=dropout):
        o = Conv1D(filters, factor * kernel_size - 1, padding=padding)(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation(activation)(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        return o

    def resnet(i):
        o = resnet_module(i, 3, activation='relu')
        o = resnet_module(o, 2, activation='relu')
        o = resnet_module(o, 1, dropout=0.0)
        if o.shape[-1] != i.shape[-1]:
            i = Conv1D(o.shape[-1], 1, padding=padding)(i)
        if use_batch_norm:
            i = BatchNormalization()(i)
        o = Add()([i, o])
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
        use_batch_norm=False):
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
         use_batch_norm=False,
         return_sequences=False):
    def conv(i):
        o = TimeDense(filters)(i)
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
           recept_field,
           pool_size,
           kernel_sizes=(7, 9, 11),
           padding='causal',
           return_sequences=False):
    def rocket(i):
        outs = []
        for kernel_size in kernel_sizes:
            dilations = [2**n for n in range(10) if 2**n * kernel_size <= recept_field]
            filters_ = int(100 * filters / len(dilations) / len(kernel_sizes))
            for dilation in dilations:
                o = Conv1D(filters_, kernel_size, padding=padding, dilation_rate=dilation, trainable=False,
                            kernel_initializer=RandomNormal(stddev=1), bias_initializer=RandomUniform(-1, 1))(i)
                outs.append(o)
        o = Concatenate()(outs)
        if return_sequences:
            o_max = CausalMaxPooling1D(pool_size)(o)
            o_avg = CausalAveragePooling1D(pool_size)(o)
        else:
            o_max = GlobalMaxPooling1D()(o)
            o_avg = GlobalAveragePooling1D()(o)
        o = Concatenate()([o_max, o_avg])
        return o
    return rocket


def Inception(filters,
              kernel_size,
              pool_size,
              padding='causal',
              dropout=0.0,
              return_sequences=False,
              use_batch_norm=False,
              bottleneck_size=32):
    def inception_module(i, activation=None, dropout=dropout):
        kernel_sizes = [kernel_size * (2 ** i) for i in range(3)]
        if bottleneck_size > 0 and int(i.shape[-1]) > 4 * bottleneck_size:
            i = Conv1D(bottleneck_size, 1, padding=padding, use_bias=not use_batch_norm)(i)
        filters_per_conv = filters // (len(kernel_sizes) + 1)
        conv_list = [Conv1D(filters_per_conv, 1, padding=padding,
                        use_bias=not use_batch_norm)(CausalMaxPooling1D(3)(i))]
        for ks in kernel_sizes:
            o = Conv1D(filters_per_conv, ks, padding=padding, use_bias=not use_batch_norm)(i)
            conv_list.append(o)
        o = Concatenate()(conv_list)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation(activation)(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        return o

    def inception(i):
        o = inception_module(i, activation='relu')
        o = inception_module(o, activation='relu')
        o = inception_module(o, dropout=0.0)
        if o.shape[-1] != i.shape[-1]:
            i = Conv1D(o.shape[-1], 1, padding=padding, use_bias=not use_batch_norm)(i)
        if use_batch_norm:
            i = BatchNormalization()(i)
        o = Add()([i, o])
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
        recept_field,
        kernel_size,
        pool_size,
        padding='causal',
        use_skip_conn=False,
        dropout=0.0,
        return_sequences=False,
        use_batch_norm=False):
    def _tcn(i):
        dilations = [2**n for n in range(10) if 2**n * kernel_size <= recept_field]
        o = tcn.TCN(filters, kernel_size, 1, dilations=dilations, padding=padding, dropout_rate=dropout,
                use_skip_connections=use_skip_conn, use_batch_norm=use_batch_norm, activation='relu', return_sequences=True)(i)
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
        o = eval(layer)(hidden_size, dropout=dropout, return_sequences=return_sequences)(i)
        if use_skip_conn and o.shape[-1] != i.shape[-1]:
            o = Add()([i, o])
        return o
    return rnn


def AHLN(hidden_size,
         recept_field,
         kernel_size,
         padding='causal',
         dropout=0.0,
         return_sequences=True,
         use_batch_norm=False,
         use_skip_conn=False):
    def ahln(i):
        o = Conv1D(hidden_size, kernel_size, padding=padding)(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        o = Conv1D(hidden_size, kernel_size, padding=padding)(o)
        if use_batch_norm:
            o = BatchNormalization()(o)
        if use_skip_conn:
            if o.shape[-1] != i.shape[-1]:
                i = TimeDense(o.shape[-1])(i)
            if use_batch_norm:
                i = BatchNormalization()(i)
            o = Add()([i, o])
        o = Activation('relu')(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        pool_list = [o]
        for pool_size in [4**n for n in range(10) if 4**n <= recept_field]:
            pool_list.append(CausalAveragePooling1D(pool_size)(o))
            pool_list.append(CausalMaxPooling1D(pool_size)(o))
            pool_list.append(CausalMinPooling1D(pool_size)(o))
        o = Concatenate()(pool_list)
        if not return_sequences:
            o = GlobalAveragePooling1D()(o)
        return o
    return ahln


def CausalAveragePooling1D(pool_size):
    def pool(i):
        if pool_size > 1:
            o = Lambda(lambda x: K.temporal_padding(x, (pool_size - 1, 0)))(i)
            o = AveragePooling1D(pool_size, strides=1, padding='valid')(o)
            scale = tf.reshape(pool_size / tf.range(1, pool_size+1, dtype=o.dtype), (1, pool_size, 1))
            o = tf.concat([scale * o[:, :pool_size, :], o[:, pool_size:, :]], axis=1)
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


def TimeDense(units, activation=None):
    def timedense(i):
        if 'ngraph_bridge' in sys.modules and len(i.shape) == 3:
            return Conv1D(units, 1, padding='causal', activation=activation)(i)
        else:
            return Dense(units, activation=activation)(i)
    return timedense


###################################################################################################
# custom loss functions

def pnl(y_true, y_pred, c=args.commission, λ=args.pnl_scale):
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
def loss_augmented_inference(r, z, λ=args.pnl_scale, c=args.commission, ϵ=args.eta, η=args.close_thresh):
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


def direct(y_true, y_pred, η=args.close_thresh):
    def score(z, y):
        a = K.sign(y - 2)
        b = K.abs(y - 2) / 2
        c = a * (b * z + 1 - b)
        return K.maximum(c, (η + 1) / 2)
    s1 = score(y_pred[:, :, 1], y_true[:, :, 1])
    s2 = score(y_pred[:, :, 2], y_true[:, :, 2])
    return s1 - s2


@jit(nopython=True)
def direct_loss(r, z, λ=args.pnl_scale, c=args.commission, η=args.close_thresh):
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

@jit(nopython=True)
def nan_to_num(x):
    for i in range(x.shape[0]):
        xi = x[i]
        if np.isnan(xi):
            x[i] = 0
        elif xi < -5:
            x[i] = -5
        elif xi > 5:
            x[i] = 5
    return x

class JLSequence(Sequence):

    def __init__(self, logger, data_path, pred_path, sequence_size, batch_size, prefetch, feature_name, label_name, weight_name, pred_name, loss, **kwargs):
        self.sess = tf.compat.v1.keras.backend.get_session()
        if not os.path.isfile(data_path):
            F, N, T = 30, 3000, 4802
            with h5py.File(data_path, 'w') as fid:
                x = np.random.randn(T, N, F).astype('float32')
                y = np.mean(x, axis=2)
                fid.create_dataset(feature_name, data=x, chunks=(T // 2, 2 * batch_size, F), **hdf5plugin.Blosc('zstd'))
                fid.create_dataset(label_name, data=y, chunks=(T // 2, 2 * batch_size), **hdf5plugin.Blosc('zstd'))
        with h5py.File(data_path, 'r') as fid:
            self.xshape = fid[feature_name].shape
            self.x = None
            self.y = fid[label_name] if label_name in fid.keys() else None
            self.yshape = (*self.xshape[:2], 0) if self.y is None else self.y.shape
            self.w = fid[weight_name] if weight_name in fid.keys() else None
            self.pred = None
        self.logger = logger
        self.data_path = data_path
        self.pred_path = pred_path
        if sequence_size == 0:
            sequence_size = self.xshape[0]
        self.sequence_size = sequence_size
        self.batch_size = batch_size
        self.prefetch = prefetch
        self.feature_name = feature_name
        self.label_name = label_name
        self.weight_name = weight_name
        self.pred_name = pred_name
        self.loss = loss
        self.n_sequences = math.floor(self.xshape[0] / self.sequence_size)
        self.n_batches = math.ceil(self.xshape[1] / batch_size)
        self.start = 0
        self.end = self.n_sequences * self.n_batches
        self.out_seq = self.yshape[0] == self.xshape[0]
        print('JLSequence shape: (', self.n_batches, 'x', self.batch_size, ',', self.n_sequences, 'x', self.sequence_size, ')')
        if args.output_dim < 1:
            if self.out_seq:
                self.output_dim = self.yshape[-1] if len(self.yshape) == 3 else 1
            else:
                self.output_dim = self.yshape[-1] if len(self.yshape) == 2 else 1
        else:
            self.output_dim = args.output_dim

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        idx = idx + self.start
        n, t = idx % self.n_batches, idx // self.n_batches
        ts = slice(self.sequence_size * t, self.sequence_size * (t + 1))
        ns = slice(self.batch_size * n, self.batch_size * (n + 1))
        ns = slice(ns.start, min(ns.stop, self.xshape[1]))
        if self.x is None:
            fid = h5py.File(self.data_path, 'r', rdcc_nbytes=1024**3, rdcc_nslots=100000)
            self.x = fid[self.feature_name]
            self.y = fid[self.label_name] if self.label_name in fid.keys() else None
            self.w = fid[self.weight_name] if self.weight_name in fid.keys() else None
            if self.prefetch:
                self.x = self.x[()]
                self.y = self.y[()] if self.y else None
                self.w = self.w[()] if self.w else None
        x = self.x[ts, ns].swapaxes(0, 1)
        if x.dtype == 'uint8':
            x = x / 128 - 1
        elif x.dtype.kind == 'f':
            nan_to_num(x.reshape(-1))
        if self.y is not None:
            if self.y.shape[0] == self.xshape[0]:
                y = self.y[ts, ns].swapaxes(0, 1)
                y = y.reshape(*y.shape[:2], -1)
            else:
                y = self.y[ns]
                y = y.reshape(*y.shape[:1], -1)
        else:
            y = None
        if self.w is not None:
            if self.w.shape[0] == self.xshape[0]:
                w = self.w[ts, ns].swapaxes(0, 1)
            else:
                w = self.w[ns]
        else:
            w = None
        if w is not None and self.loss in  ('pnl', 'direct'):
            w = w.reshape(*w.shape, 1)
            y = np.multiply(y, w)
            w = None
        if self.loss == 'direct' and y is not None:
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
        if pred.ndim < 3:
            fid = h5py.File(self.pred_path, 'w')
            fid.create_dataset(self.pred_name, data=pred)
            return
        npred = 0
        for idx in range(len(self)):
            idx = idx + self.start
            n, t = idx % self.n_batches, idx // self.n_batches
            ts = slice(self.sequence_size * t, self.sequence_size * (t + 1))
            ns = slice(self.batch_size * n, self.batch_size * (n + 1))
            ns = range(ns.start, min(ns.stop, self.xshape[1]))
            y = pred[npred:(npred + len(ns)), :, :]
            shape = (*self.xshape[:2], y.shape[-1])
            if self.pred is None:
                self.fid_pred = h5py.File(self.pred_path, 'w')
                self.fid_pred.create_dataset(self.pred_name, shape, dtype='float32')
                self.pred = self.fid_pred[self.pred_name]
            self.pred[ts, ns, :] = y.swapaxes(0, 1)
            self.fid_pred.flush()
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
if not args.test and usegpu and gpus:
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
gen = JLSequence(logger, **vars(args))
trn_gen, val_gen = gen.split(args.validation_split)

###################################################################################################
# model testing

if args.test:
    model = load_model(args.model_path, compile=False)
    gen.fill_pred(model.predict(gen))
    exit()

###################################################################################################
# model building

hidden_sizes = list(map(int, args.hidden_sizes.split(',')))
args.pool_size = min(gen.xshape[0] // 3, args.pool_size)
loss, out_activation = args.loss, args.out_activation
loss = 'binary_crossentropy' if loss == 'bce' else loss
loss = 'categorical_crossentropy' if loss == 'cce' else loss
loss = 'sparse_categorical_crossentropy' if loss == 'spcce' else loss
out_activation = 'sigmoid' if loss == 'binary_crossentropy' else args.out_activation
out_activation = 'softmax' if 'categorical_crossentropy' in loss else out_activation
out_activation = 'tanh' if loss == 'pnl' else out_activation

# expected input batch shape: (N, T, F)
if len(gen.xshape) == 3:
    o = i = Input((None, gen.xshape[2]))
else:
    o = i = Input((None,))
    o = Embedding(args.input_dim, args.embed_dim)(o)
if os.getenv('USE_TFLITE', '0') == '1' or args.layer == 'MLP':
    i.set_shape((args.batch_size, *i.shape[1:]))
    if args.layer == 'MLP':
        o = Flatten()(o)
for (l, h) in enumerate(hidden_sizes):
    return_sequences = l + 1 < len(hidden_sizes) or gen.out_seq
    if args.layer == 'MLP':
        o = MLP(h, dropout=args.dropout, use_batch_norm=args.use_batch_norm)(o)
    elif args.layer == 'Conv':
        o = Conv(h, dropout=args.dropout, use_batch_norm=args.use_batch_norm,
                 return_sequences=return_sequences)(o)
    elif args.layer == 'ResNet':
        o = ResNet(h, args.kernel_size, args.pool_size, dropout=args.dropout,
                   use_batch_norm=args.use_batch_norm, return_sequences=return_sequences)(o)
    elif args.layer == 'Inception':
        o = Inception(h, args.kernel_size, args.pool_size, dropout=args.dropout, use_batch_norm=args.use_batch_norm,
                      return_sequences=return_sequences, bottleneck_size=args.bottleneck_size)(o)
    elif args.layer == 'Rocket':
        if l == 0:
            kernel_sizes = list(map(int, args.kernel_sizes.split(',')))
            o = Rocket(h, args.recept_field, args.pool_size, kernel_sizes, return_sequences=return_sequences)(o)
        else:
            o = TimeDense(h, activation='relu')(o)
    elif args.layer == 'TCN':
        o = TCN(h, args.recept_field, args.kernel_size, args.pool_size, dropout=args.dropout, use_skip_conn=args.use_skip_conn,
                use_batch_norm=args.use_batch_norm, return_sequences=return_sequences)(o)
    elif args.layer == 'AHLN':
        o = AHLN(h, args.recept_field, args.kernel_size, dropout=args.dropout, use_batch_norm=args.use_batch_norm,
                 use_skip_conn=args.use_skip_conn, return_sequences=return_sequences)(o)
    else:
        o = ResRNN(h, dropout=args.dropout, return_sequences=return_sequences,
                   use_skip_conn=args.use_skip_conn, layer=args.layer)(o)
o = Activation(out_activation, dtype='float')(TimeDense(gen.output_dim)(o))
if loss == 'direct':
    o = Concatenate()([o, o, o])
model = Model(inputs=[i], outputs=[o])
print(model.summary())

# warm start
resume_from_epoch = 0
if args.warm_start:
    for try_epoch in range(args.epochs, 0, -1):
        h5 = args.log_dir + '/' + args.ckpt_fmt.format(epoch=try_epoch)
        if os.path.isfile(h5):
            resume_from_epoch = try_epoch
            model = load_model(h5, compile=False)
            print('warm start...')
            break
else:
    import shutil
    shutil.rmtree(args.log_dir, ignore_errors=True)
if args.reset_epoch:
    resume_from_epoch = 0

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
        args.warmup_epochs, verbose=1))
if loss == 'direct':
    callbacks.append(logger)
if local_rank == 0:
    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    callbacks.append(AltModelCheckpoint(args.log_dir + '/' + args.ckpt_fmt, base_model))
    callbacks.append(TensorBoard(args.log_dir))
    if args.validation_split > 0:
        callbacks.append(EarlyStopping(patience=args.patience, verbose=1))
    else:
        callbacks.append(EarlyStopping('loss', patience=args.patience, verbose=1))
    if args.factor < 1:
        callbacks.append(ReduceLROnPlateau(
            'loss', args.factor, args.patience, min_lr=1e-6, varbose=1))

# optimizer
weight_decays = {l: args.l2 for l in get_weight_decays(model)}
if base_model.optimizer is None:
    # Horovod: adjust learning rate based on number of GPUs.
    if args.optimizer == 'SGDW':
        opt = SGDW(10 * args.lr * world_size, momentum = 0.9, nesterov=True, weight_decays=weight_decays)
    elif args.optimizer == 'AdamW':
        opt = AdamW(args.lr * world_size, weight_decays=weight_decays)
    else:
        opt = eval(args.optimizer)(args.lr * world_size)
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
if args.debug:
    model.run_eagerly = True

###################################################################################################
# model building

# lr finder
if args.lr == 0:
    lr_finder = LRFinder(model)
    epochs_ = max(1, 100 * args.batch_size // N)
    lr_finder.find_generator(gen, 1e-6, 1e-2, epochs_)
    best_lr = min(1e-3, lr_finder.get_best_lr(5))
    K.set_value(model.optimizer.lr, best_lr)
    print(40 * '=', '\nSet lr to: %s\n' % best_lr,  40 * '=')

# train model
model.fit(
    x=trn_gen,
    epochs=args.epochs,
    verbose=1 if local_rank == 0 else 0,
    callbacks=callbacks,
    validation_data=val_gen if len(val_gen) > 1 else None,
    shuffle=True,
    initial_epoch=resume_from_epoch,
    steps_per_epoch=len(trn_gen) // world_size,
    validation_steps=len(val_gen) // world_size,
    workers=0 if loss == 'direct' else 4,
    use_multiprocessing=args.use_multiprocessing
)
base_model.save(args.model_path)
