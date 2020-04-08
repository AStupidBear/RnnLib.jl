# convert model
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_KERAS'] = '1'

import numpy as np
import onnxmltools
import onnxruntime as ort
import tcn
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.platform import gfile

from ind_rnn import IndRNN

# load model
custom_objects = {'TCN': tcn.TCN, 'IndRNN': IndRNN}
model = load_model('rnn.h5', compile=False, custom_objects=custom_objects)
input_shape = model.inputs[0].shape.as_list()
input_shape[0] = input_shape[0] if input_shape[0] else 32
input_shape[1] = input_shape[1] if input_shape[1] else 666
x = np.random.randn(*input_shape).astype('float32')

# keras inference
model(x, training=False)
ti = time.time()
model(x, training=False)
print('keras time: ', time.time() - ti)

# tf inference
keras2tf = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'keras2tf.py')
os.system('CUDA_VISIBLE_DEVICES=-1 python %s --input_model=rnn.h5 --output_model=rnn.pb --output_nodes_prefix=output_ 2> /dev/null' % keras2tf)
os.system('CUDA_VISIBLE_DEVICES=-1 python -m tensorflow.python.tools.optimize_for_inference --input=rnn.pb --output=rnn.pb --input_names=input_1 --output_names=output_1 2> /dev/null')
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    with gfile.GFile('rnn.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tensor_input, tensor_output = tf.import_graph_def(
            graph_def, return_elements=['input_1:0', 'output_1:0'])
    sess.run(tensor_output, {tensor_input: x})
    ti = time.time()
    sess.run(tensor_output, {tensor_input: x})
    print('tf time: ', time.time() - ti)

# tflite inference
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    open('rnn.tflite', 'wb').write(converter.convert())
    interpreter = tf.lite.Interpreter('rnn.tflite')
    interpreter.allocate_tensors()
    input_tensor = interpreter.tensor(
        interpreter.get_input_details()[0]['index'])
    output_tensor = interpreter.tensor(
        interpreter.get_output_details()[0]['index'])
    interpreter.invoke()
    ti = time.time()
    interpreter.invoke()
    print('tflite time: ', time.time() - ti)
except Exception as e:
    pass

# onnx inference
try:
    onnx_model = onnxmltools.convert_keras(model)
    onnxmltools.utils.save_model(onnx_model, 'rnn.onnx')
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession('rnn.onnx', so)
    input_name = sess.get_inputs()[0].name
    for provider in sess.get_providers():
        sess.set_providers([provider])
        sess.run(None, {input_name: x})[0]
        ti = time.time()
        sess.run(None, {input_name: x})[0]
        print('onnx-' + provider, 'time: ', time.time() - ti)
except Exception as e:
    pass

# # ngraph inference
# import onnx
# import ngraph as ng
# from ngraph_onnx.onnx_importer.importer import import_onnx_model
# ng_func = import_onnx_model(onnx.load('rnn.onnx'))
# ngrt = ng.runtime(backend_name='CPU')
# ng_comp = ngrt.computation(ng_func)
# ng_comp(x)

# # caffe2 inference
# import onnx
# from caffe2.python.onnx.backend import run_model
# model = onnx.load('rnn.onnx')
# outputs = run_model(model, [x])
