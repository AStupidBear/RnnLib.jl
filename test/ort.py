import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_KERAS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"

import keras
import numpy as np
import onnxmltools
import onnxruntime as ort
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv1D, Dense

x = np.random.randn(128, 1000, 30).astype('float32')
y = np.random.randn(128, 1000, 1).astype('float32')

o = i = Input((1000, x.shape[-1]), batch_size=128)
for n in range(5):
    o = Conv1D(64, 3, activation='relu', padding='same')(o)
o = Conv1D(1, 3, padding='same')(o)
model = Model(inputs=[i], outputs=[o])
model.compile(loss='mse', optimizer='sgd')

model.fit(x, y, epochs=1, batch_size=128)
onnx_model = onnxmltools.convert_keras(model, target_opset=11)
onnxmltools.utils.save_model(onnx_model, 'model.onnx')

model.predict(x, batch_size=128)
ti = time.time()
model.predict(x, batch_size=128)
print('keras time: ', time.time() - ti)

so = ort.SessionOptions()
sess = ort.InferenceSession("model.onnx")
sess.set_providers(['DnnlExecutionProvider'])
input_name = sess.get_inputs()[0].name
sess.run(None, {input_name: x})[0]
ti = time.time()
sess.run(None, {input_name: x})[0]
print('onnx time: ', time.time() - ti)
