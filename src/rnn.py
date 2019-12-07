#!/usr/bin/env python
import time
from tcn import TCN
from keras.models import model_from_json, load_model, Input, Model
from keras.layers import Dense, Conv1D, GRU, CuDNNGRU, LSTM, CuDNNLSTM
from keras.layers import SpatialDropout1D, GlobalAveragePooling1D, BatchNormalization
from keras.layers import Layer, add, Lambda, Flatten, TimeDistributed, Activation
from keras.utils import multi_gpu_model
from keras.utils.io_utils import HDF5Matrix
from keras import backend as K
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
parser.add_argument('--layer', type=str, default='TCN')
parser.add_argument('--out_activation', type=str, default='linear')
parser.add_argument('--hidden_sizes', type=str, default='10,10')
parser.add_argument('--loss', type=str, default='pnl')
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--dilations', type=str, default='1,2,4,8,16,32,64')
parser.add_argument('--l2', type=float, default=0)
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--use_batch_norm', type=int, default=0)
parser.add_argument('--commission', type=float, default=0)
parser.add_argument('--pnl_scale', type=float, default=224)
parser.add_argument('--out_dim', type=int, default=0)
parser.add_argument('--validation_split', type=float, default=0.3)
parser.add_argument('--patience', type=int, default=10)
args = parser.parse_args()
data, file, warm_start, test = args.data, args.file, args.warm_start, args.test
lr, batch_size, epochs, layer = args.lr, args.batch_size, args.epochs, args.layer
out_activation, hidden_sizes = args.out_activation, args.hidden_sizes
loss, kernel_size, dilations = args.loss, args.kernel_size, args.dilations
l2, dropout_rate, use_batch_norm = args.l2, args.dropout_rate, args.use_batch_norm
commission, pnl_scale, out_dim = args.commission, args.pnl_scale, args.out_dim
validation_split, patience = args.validation_split, args.patience
hidden_sizes = list(map(int, hidden_sizes.split(',')))
dilations = list(map(int, dilations.split(',')))
loss = 'binary_crossentropy' if loss == 'bce' else loss
loss = 'categorical_crossentropy' if loss == 'cce' else loss
loss = 'sparse_categorical_crossentropy' if loss == 'spcce' else loss
out_activation = 'sigmoid' if loss == 'binary_crossentropy' else out_activation
out_activation = 'softmax' if 'categorical_crossentropy' in loss else out_activation
out_activation = 'tanh' if loss == 'pnl' else out_activation

# custom functions

def ResNet(filters,
        kernel_size,
        padding='causal',
        dropout_rate=0.0,
        return_sequences=False,
        activation='relu',
        use_batch_norm=False):
    def resnet(i):
        o = Conv1D(filters, 4 * kernel_size, padding=padding)(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        if dropout_rate > 0:
            o = SpatialDropout1D(dropout_rate)(o)
        o = Conv1D(filters, 2 * kernel_size, padding=padding)(o)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        if dropout_rate > 0:
            o = SpatialDropout1D(dropout_rate)(o)
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
        if activation != 'linear' and dropout_rate > 0:
            o = SpatialDropout1D(dropout_rate)(o)
        if not return_sequences:
            o = GlobalAveragePooling1D()(o)
        return o
    return resnet


def MLP(hidden_size,
        dropout_rate=0.0,
        activation='relu',
        use_batch_norm=False):
    def mlp(i):
        o = Dense(hidden_size)(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation(activation)(o)
        if activation != 'linear' and dropout_rate > 0:
            o = SpatialDropout1D(dropout_rate)(o)
        return o
    return mlp


def Conv(filters,
        dropout_rate=0.0,
        activation='relu',
        use_batch_norm=False,
        return_sequences=False):
    def conv(i):
        o = Conv1D(filters, 1, padding='causal')(i)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation(activation)(o)
        if activation != 'linear' and dropout_rate > 0:
            o = SpatialDropout1D(dropout_rate)(o)
        if not return_sequences:
            o = Lambda(lambda x: x[:, -1, :])(o)
        return o
    return conv


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


def add_weight_decay(model, weight_decay):
    if (weight_decay is None) or (weight_decay == 0.0):
        return
    # recursion inside the model
    def add_decay_loss(m, factor):
        if isinstance(m, keras.Model):
            for layer in m.layers:
                add_decay_loss(layer, factor)
        else:
            for param in m.trainable_weights:
                with keras.backend.name_scope('weight_regularizer'):
                    if 'bias' not in param.name:
                        regularizer = keras.regularizers.l2(factor)(param)
                        m.add_loss(regularizer)
    # weight decay and l2 regularization differs by a factor of 2
    add_decay_loss(model, weight_decay / 2.0)
    return

# f16 precision
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
if layer == "MLP":
    o = i = Input(shape=(T, F))
    o = Flatten()(o)
else:
    o = i = Input(shape=(None, F))
if layer == 'TCN':
    o = TCN(hidden_sizes[0], kernel_size, 1, dilations=dilations, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm, activation='relu', return_sequences=out_seq)(o)
for (l, h) in enumerate(hidden_sizes):
    return_sequences = l + 1 < len(hidden_sizes) or out_seq
    if layer == 'ResNet':
        o = ResNet(h, kernel_size, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm, return_sequences=return_sequences)(o)
    elif layer == 'MLP':
        o = MLP(h, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)(o)
    elif layer == "Conv":
        o = Conv(h, dropout_rate=dropout_rate, use_batch_norm=use_batch_norm, return_sequences=return_sequences)(o)
    elif isrnn(layer):
        o = eval(layer)(h, dropout=dropout_rate, return_sequences=return_sequences)(o)
o = Dense(out_dim, activation=out_activation)(o)
model = Model(inputs=[i], outputs=[o])
add_weight_decay(model, l2)
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
if use_horovod:
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    callbacks = callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    # Horovod: adjust learning rate based on number of GPUs.
    opt = keras.optimizers.Adam(lr * world_size, clipnorm=0.5)
    opt = hvd.DistributedOptimizer(opt)
else:
    callbacks = [keras.callbacks.ModelCheckpoint(file)]
    opt = keras.optimizers.Adam(lr, clipnorm=1)
if local_rank == 0:
    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    callbacks.append(keras.callbacks.ModelCheckpoint(file))
    if validation_split > 0:
        callbacks.append(keras.callbacks.EarlyStopping(patience=patience, verbose=1))
    # callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-4))

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
