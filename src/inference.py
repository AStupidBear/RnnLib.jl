import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import tensorflow as tf
from tensorflow.python.platform import gfile

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

def get_predict_function(name):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    with gfile.GFile(name + '.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tensor_input, tensor_output = tf.import_graph_def(
            graph_def, return_elements=['input_1:0', 'output_1:0'])
    def predict(x):
        return sess.run(tensor_output, {tensor_input: x})
    return predict

# import onnx
# import onnxruntime as rt

# model = onnx.load("rnn.onnx")
# siz = 0
# for tensor in model.graph.initializer:
#     data = tensor.float_data
#     siz += len(data)
#     for i in range(len(data)):
#         data[i] = data[i]
# onnx.save_model(model, "rnn.onnx")

# sess = rt.InferenceSession("rnn.onnx")
# input_name = sess.get_inputs()[0].name
# input_shape = sess.get_inputs()[0].shape
# y = sess.run(None, {input_name: x})[0]

# all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]