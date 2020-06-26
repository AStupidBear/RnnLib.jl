import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_KERAS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import onnxmltools
import onnxruntime as ort
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.python.platform import gfile

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

x = np.random.randn(32, 256, 256, 30).astype('float32')
y = np.random.randn(32, 256, 256, 1).astype('float32')

o = i = Input(x.shape[1:], batch_size=32)
for n in range(5):
    o = Conv2D(64, (3, 3), activation='relu', padding='same')(o)
o = Conv2D(1, (3, 3), padding='same')(o)
model = Model(inputs=[i], outputs=[o])
model.compile(loss='mse', optimizer='sgd')

model.save('model.h5')
onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, 'model.onnx')

model(x, training=False)
ti = time.time()
model(x, training=False)
print('keras time: ', time.time() - ti)

keras2tf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/keras2tf.py')
if os.path.isfile(keras2tf):
    os.system('python %s --input_model=model.h5 --output_model=model.pb --output_nodes_prefix=output_' % keras2tf)
    os.system('python -m tensorflow.python.tools.optimize_for_inference --input=model.pb --output=model.pb --input_names=input_1 --output_names=output_1')
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        with gfile.GFile('model.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tensor_input, tensor_output = tf.import_graph_def(
                graph_def, return_elements=['input_1:0', 'output_1:0'])
        sess.run(tensor_output, {tensor_input: x})
        ti = time.time()
        sess.run(tensor_output, {tensor_input: x})
        print('tf time: ', time.time() - ti)

so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess = ort.InferenceSession("model.onnx", so)
for provider in sess.get_providers():
    sess.set_providers([provider])
    input_name = sess.get_inputs()[0].name
    sess.run(None, {input_name: x})[0]
    ti = time.time()
    sess.run(None, {input_name: x})[0]
    print('onnx-' + provider, 'time: ', time.time() - ti)
