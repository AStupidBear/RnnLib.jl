import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_KERAS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import onnxmltools
import onnxruntime as ort
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, Dense

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

x = np.random.randn(128, 1000, 30).astype('float32')
y = np.random.randn(128, 1000, 1).astype('float32')

o = i = Input((1000, x.shape[-1]), batch_size=128)
for n in range(5):
    o = Conv1D(64, 3, activation='relu', padding='same')(o)
o = Conv1D(1, 3, padding='same')(o)
model = Model(inputs=[i], outputs=[o])
model.compile(loss='mse', optimizer='sgd')

model.fit(x, y, epochs=1, batch_size=128)
model.save('rnn.h5')
onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, 'rnn.onnx')

model(x, training=False)
ti = time.time()
model(x, training=False)
print('keras time: ', time.time() - ti)

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess = ort.InferenceSession("rnn.onnx", so)
for provider in sess.get_providers():
    sess.set_providers([provider])
    input_name = sess.get_inputs()[0].name
    sess.run(None, {input_name: x})[0]
    ti = time.time()
    sess.run(None, {input_name: x})[0]
    print('onnx-' + provider, 'time: ', time.time() - ti)