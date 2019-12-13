#!/usr/bin/env python
from warnings import filterwarnings
filterwarnings("ignore", module='numpy')
filterwarnings("ignore", module='tensorflow')

import time
from tcn import TCN as _TCN
from keras.models import model_from_json, load_model, Input, Model
from keras.layers import Layer, Lambda, Flatten, Activation, add, concatenate
from keras.layers import Dense, Conv1D, Dropout, SpatialDropout1D, BatchNormalization
from keras.layers import AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import GRU, CuDNNGRU, LSTM, CuDNNLSTM, TimeDistributed
from keras.initializers import RandomNormal, RandomUniform
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import multi_gpu_model
from keras.utils.io_utils import HDF5Matrix
from keras import backend as K
from keras_adamw.optimizers_225 import AdamW
from keras_adamw.utils import get_weight_decays
import tensorflow as tf
import keras
import numpy as np
import onnxmltools
import logging
import gc
import sys
import math
import os
import argparse

print('current path %s\n' % os.getcwd())
# parse args
parser = argparse.ArgumentParser(description='distributed rnn regressor')
parser.add_argument('--data', type=str, default='train.rnn')
parser.add_argument('--file', type=str, default='rnn.h5')
parser.add_argument('--warm_start', type=int, default=0)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--layer', type=str, default='Inception')
parser.add_argument('--out_activation', type=str, default='linear')
parser.add_argument('--hidden_sizes', type=str, default='128')
parser.add_argument('--loss', type=str, default='pnl')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--kernel_sizes', type=str, default='7,9,11')
parser.add_argument('--pool_size', type=int, default=1)
parser.add_argument('--max_dilation', type=int, default=64)
parser.add_argument('--l2', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--use_batch_norm', type=int, default=1)
parser.add_argument('--bottleneck_size', type=int, default=32)
parser.add_argument('--commission', type=float, default=0)
parser.add_argument('--pnl_scale', type=float, default=224)
parser.add_argument('--out_dim', type=int, default=0)
parser.add_argument('--validation_split', type=float, default=0.2)
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()
data, file, warm_start, test = args.data, args.file, args.warm_start, args.test
lr, batch_size, epochs, layer = args.lr, args.batch_size, args.epochs, args.layer
out_activation, loss, kernel_size = args.out_activation, args.loss, args.kernel_size
pool_size, max_dilation, dropout = args.pool_size, args.max_dilation, args.dropout
l2, use_batch_norm, bottleneck_size = args.lr, args.use_batch_norm, args.bottleneck_size
commission, pnl_scale, out_dim = args.commission, args.pnl_scale, args.out_dim
validation_split, patience = args.validation_split, args.patience
hidden_sizes = list(map(int, args.hidden_sizes.split(',')))
kernel_sizes = list(map(int, args.kernel_sizes.split(',')))

loss = 'binary_crossentropy' if loss == 'bce' else loss
loss = 'categorical_crossentropy' if loss == 'cce' else loss
loss = 'sparse_categorical_crossentropy' if loss == 'spcce' else loss
out_activation = 'sigmoid' if loss == 'binary_crossentropy' else out_activation
out_activation = 'softmax' if 'categorical_crossentropy' in loss else out_activation
out_activation = 'tanh' if loss == 'pnl' else out_activation

# custom functions

def ResNet(filters,
        kernel_size,
        pool_size,
        padding='causal',
        dropout=0.0,
        return_sequences=False,
        activation='relu',
        use_batch_norm=True):
    def resnet(i):
        o = Conv1D(filters, 3 * kernel_size - 1, padding=padding)(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        o = Conv1D(filters, 2 * kernel_size - 1, padding=padding)(o)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        if dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        o = Conv1D(filters, kernel_size, padding=padding)(o)
        if use_batch_norm:
            o = BatchNormalization()(o)
        # expand channels for the sum
        if filters != i.shape[-1].value:
            i = Conv1D(filters, 1, padding=padding)(i)
        if use_batch_norm:
            i = BatchNormalization()(i)
        o = add([i, o])
        o = Activation(activation)(o)
        if activation != 'linear' and dropout > 0:
            o = SpatialDropout1D(dropout)(o)
        if return_sequences:
            o = CausalAveragePooling1D(pool_size)(o)
        else:
            o = GlobalAveragePooling1D()(o)
        return o
    return resnet


def MLP(hidden_size,
        dropout=0.0,
        activation='relu',
        use_batch_norm=True):
    def mlp(i):
        o = Dense(hidden_size)(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation(activation)(o)
        if activation != 'linear' and dropout > 0:
            o = Dropout(dropout)(o)
        return o
    return mlp


def Conv(filters,
        dropout=0.0,
        activation='relu',
        use_batch_norm=True,
        return_sequences=False):
    def conv(i):
        o = Conv1D(filters, 1, padding='causal')(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation(activation)(o)
        if activation != 'linear' and dropout > 0:
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
        activation='relu',
        use_batch_norm=True,
        bottleneck_size=32):
    def inception_module(i):
        kernel_sizes = [kernel_size * (2 ** i) for i in range(3)]
        conv_list = [Conv1D(filters // 4, 1, padding=padding, use_bias=False)(CausalMaxPooling1D(3)(i))]
        if bottleneck_size > 0 and int(i.shape[-1]) > 4 * bottleneck_size:
            i = Conv1D(bottleneck_size, 1, padding=padding, use_bias=False)(i)
        for ks in kernel_sizes:
            o = Conv1D(filters // 4, ks, padding=padding, use_bias=False)(i)
            conv_list.append(o)
        o = concatenate(conv_list)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation(activation)(o)
        return o
    def inception(i):
        o = inception_module(i)
        o = inception_module(o)
        o = inception_module(o)
        i = Conv1D(int(o.shape[-1]), 1, padding=padding, use_bias=False)(i)
        if use_batch_norm:
            i = BatchNormalization()(i)
        o = add([i, o])
        o = Activation(activation)(o)
        if activation != 'linear' and dropout > 0:
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
    def tcn(i):
        dilations = [2**n for n in range(10) if 2**n <= max_dilation]
        o = _TCN(filters, kernel_size, 1, dilations=dilations, padding=padding, dropout_rate=dropout, 
                use_batch_norm=use_batch_norm, activation=activation, return_sequences=True)(i)
        if return_sequences:
            o = CausalAveragePooling1D(pool_size)(o)
        else:
            o = GlobalAveragePooling1D()(o)
        return o
    return tcn


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


def isrnn(layer):
    return layer in ['GRU', 'CuDNNGRU', 'LSTM', 'CuDNNLSTM']


def pnl(y_true, y_pred, c=commission, λ=pnl_scale):
    r, p, c = λ * y_true, y_pred, λ * c
    l = - K.mean(r * p)
    if not out_seq:
        return l
    c1 = c / K.cast(K.shape(p)[1], 'float32')
    if c > 0:
        if T > 1:
            Δp = p[:, 1:, :] - p[:, :-1, :]
            l += c * K.mean(K.abs(Δp))
        l += c1 * K.mean(K.abs(p[:, 0, :]))
        l += c1 * K.mean(K.abs(p[:, -1, :]))
    return l

# # f16 precision
# if not use_batch_norm:
#     K.set_floatx('float16')
#     K.set_epsilon(1e-4)

# setenv
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
usegpu = os.getenv('USE_GPU', '1') == '1'
if not usegpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# set pnl loss
keras.losses.pnl = pnl

# horovod init
try:
    import horovod.keras as hvd
    hvd.init()
    world_size = hvd.size()
    use_horovod = world_size > 1
    local_rank = hvd.local_rank()
    # Horovod: adjust number of epochs based on number of GPUs.
    epochs = int(math.ceil(epochs / world_size))
except:
    world_size = 1
    use_horovod = False
    local_rank = 0

# load hdf5 data or generate dummy data
if os.path.isfile(data) and os.name != 'nt':
    x = HDF5Matrix(data, 'x')
    N, T, F = x.shape
    if isrnn(layer):
        step = math.floor(N / batch_size / world_size) * batch_size
    else:
        step = math.floor(N / world_size)
    start = local_rank * step
    end = start + step
    x = HDF5Matrix(data, 'x', start, end)
    p = HDF5Matrix(data, 'p', start, end)
    y = HDF5Matrix(data, 'y', start, end).data[()]
    w = HDF5Matrix(data, 'w', start, end).data[()]
    w_mean = HDF5Matrix(data, 'w').data[:].mean()
    N, T, F = x.shape
else:
    F, T, N = 30, 66608, 224 // world_size
    N = N // batch_size * batch_size
    x = np.random.randn(N, T, F)
    y = np.random.randn(N, T, 1)
    p = np.random.randn(N, T, 1)
    w = np.random.rand(N, T)
    w_mean = 1
out_dim = y.shape[-1] if out_dim < 1 else out_dim
out_seq = len(y.shape) == 3

# set gpu specific options and reset keras sessions
if test == 0 and usegpu and tf.test.is_gpu_available():
    K.get_session().close()
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(local_rank)
    K.set_session(tf.Session(config=config))

# test mode
if test == 1:
    model = load_model(file, compile=False)
    p.data[start:end] = model.predict(x, batch_size=batch_size)
    exit()

# Expected input batch shape: (N, T, F)
pool_size = min(T // 3, pool_size)
max_dilation = min(T // kernel_size, max_dilation)
if layer == "MLP":
    o = i = Input(shape=(T, F))
    o = Flatten()(o)
else:
    o = i = Input(shape=(None, F))
if x.dtype == 'uint8':
    o = Lambda(lambda z: z / 128 - 1)(o)
for (l, h) in enumerate(hidden_sizes):
    return_sequences = l + 1 < len(hidden_sizes) or out_seq
    if layer == 'MLP':
        o = MLP(h, dropout=dropout, use_batch_norm=use_batch_norm)(o)
    elif layer == "Conv":
        o = Conv(h, dropout=dropout, use_batch_norm=use_batch_norm, return_sequences=return_sequences)(o)
    elif layer == 'ResNet':
        o = ResNet(h, kernel_size, pool_size, dropout=dropout, use_batch_norm=use_batch_norm, return_sequences=return_sequences)(o)
    elif layer == 'Inception':
        o = Inception(h, kernel_size, pool_size, dropout=dropout, use_batch_norm=use_batch_norm, return_sequences=return_sequences, bottleneck_size=bottleneck_size)(o)
    elif layer == 'Rocket':
        if l == 0:
            o = Rocket(h, pool_size, max_dilation, kernel_sizes, return_sequences=return_sequences)(o)
        else:
            o = Dense(h, activation='relu')(o)
    elif layer == 'TCN':
        if l == 0:
            o = TCN(h, kernel_size, pool_size, max_dilation, dropout=dropout, use_batch_norm=use_batch_norm, return_sequences=return_sequences)(o)
        else:
            o = Dense(h, activation='relu')(o)
    elif isrnn(layer):
        o = eval(layer)(h, dropout=dropout, return_sequences=return_sequences)(o)
o = Dense(out_dim, activation=out_activation)(o)
model = Model(inputs=[i], outputs=[o])
print(model.summary())

# warm start
if warm_start and os.path.isfile(file):
    model = load_model(file, compile=False)

# pnl loss
if loss == 'pnl':
    w = w.reshape(*w.shape, 1)
    y = np.multiply(y, w)
    w = None
else:
    w = w / w_mean

# multi-gpu
gpu_devices = [dev.name for dev in K.get_session().list_devices()
               if ':GPU' in dev.name]
if not use_horovod and len(gpu_devices) > 1:
    model.save(file, include_optimizer=False)
    with tf.device('/cpu:0'):
        model = load_model(file, compile=False)
    pmodel = multi_gpu_model(model)
else:
    pmodel = model

# optimizer and callbacks
callbacks = []
weight_decays = {l: l2 for l in get_weight_decays(model)}
if use_horovod:
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    # Horovod: adjust learning rate based on number of GPUs.
    opt = AdamW(lr * world_size, clipnorm=1, weight_decays=weight_decays)
    opt = hvd.DistributedOptimizer(opt)
else:
    opt = AdamW(lr, clipnorm=1, weight_decays=weight_decays)
if local_rank == 0:
    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    callbacks.append(ModelCheckpoint(file))
    if validation_split > 0:
        callbacks.append(EarlyStopping(patience=patience, verbose=1))
    # callbacks.append(ReduceLROnPlateau(monitor='loss', factor=0.5, min_lr=1e-6, varbose=1))

# compile
sample_weight_mode = None if w is None or not out_seq else 'temporal'
if 'mse' in str(loss):
    metric = ['mae']
elif 'crossentropy' in str(loss):
    metric = ['acc']
    if 'binary' in str(loss):
        metric.append(pnl)
else:
    metric = None
pmodel.compile(loss=loss, optimizer=opt, metrics=metric, sample_weight_mode=sample_weight_mode)

# train model
pmodel.fit(x, y, sample_weight=w,
           batch_size=batch_size,
           callbacks=callbacks,
           epochs=epochs,
           verbose=1,
           shuffle=False,
           validation_split=validation_split)
model.save(file, include_optimizer=False)
score = model.evaluate(x, y, sample_weight=w, batch_size=batch_size)
print('training loss:', score)

# # convert keras to onnx
# onnx_model = onnxmltools.convert_keras(model)
# onnxmltools.utils.save_model(onnx_model, 'rnn.onnx')

# # onnxruntime inference
# import time
# import numpy as np
# import onnxruntime as rt
# sess = rt.InferenceSession("rnn.onnx")
# input_name = sess.get_inputs()[0].name
# input_shape = sess.get_inputs()[0].shape
# x_onnx = np.random.randn(1000, *input_shape[1:]).astype('float32')
# ti = time.time()
# y_onnx = sess.run(None, {input_name: x_onnx})[0]
# print(time.time() - ti)
# ti = time.time()
# y_keras = model.predict(x_onnx, batch_size=1000)
# print(time.time() - ti)

# # ngraph inference
# import numpy as np
# import onnx
# import ngraph as ng
# from ngraph_onnx.onnx_importer.importer import import_onnx_model
# ng_func = import_onnx_model(onnx.load('rnn.onnx'))
# rt = ng.runtime(backend_name='CPU')
# ng_comp = rt.computation(ng_func)
# x_ngraph = np.random.randn(32, 666, 30).astype('float32')
# ng_comp(x_ngraph)
