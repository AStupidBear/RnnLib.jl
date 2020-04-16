import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_KERAS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import onnxmltools
import onnxruntime as ort
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv1D,
                                     Dense, MaxPooling1D, add, concatenate)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def Inception(filters,
            kernel_size,
            pool_size,
            padding='same',
            use_batch_norm=False):
    def inception_module(i):
        kernel_sizes = [kernel_size * (2 ** i) for i in range(3)]
        pool_i = MaxPooling1D(pool_size=3, strides=1, padding=padding)(i)
        conv_list = [Conv1D(filters // 4, 1, padding=padding, use_bias=False)(pool_i)]
        for ks in kernel_sizes:
            o = Conv1D(filters // 4, ks, padding=padding, use_bias=False)(i)
            conv_list.append(o)
        o = concatenate(conv_list)
        if use_batch_norm:
            o = BatchNormalization()(o)
        o = Activation('relu')(o)
        o = MaxPooling1D(pool_size, strides=1, padding='same')(o)
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
        return o
    return inception

F, N, T = 30, 32, 4096
x = np.random.randn(N, T, F).astype('float32')
y = np.random.randn(N, T, 1).astype('float32')

filters_list = [128, 128]

for pool_size in [3, 32, 64, 128]:
    print('---------------------')
    print('pool_size', pool_size)
    print('---------------------')
    o = i = Input((T, F))
    for filters in filters_list:
        o = Inception(filters, 3, pool_size=pool_size)(o)
    o = Conv1D(1, 1, padding='same')(o)
    model = Model(inputs=[i], outputs=[o])
    model.compile(loss='mse', optimizer='sgd')

    model.save('model.h5')
    onnx_model = onnxmltools.convert_keras(model)
    onnxmltools.utils.save_model(onnx_model, 'model.onnx')

    model(x, training=False)
    ti = time.time()
    model(x, training=False)
    print('keras time: ', time.time() - ti)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession("model.onnx", so)
    for provider in sess.get_providers()[:1]:
        sess.set_providers([provider])
        input_name = sess.get_inputs()[0].name
        sess.run(None, {input_name: x})[0]
        ti = time.time()
        sess.run(None, {input_name: x})[0]
        print('onnx-' + provider, 'time: ', time.time() - ti)
