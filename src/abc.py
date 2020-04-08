import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.platform import gfile
import time
import os
from tensorflow.python.framework import ops

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

ops.reset_default_graph()
with tf.Session() as sess:
    with gfile.FastGFile('rnn_opt.pb','rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tensor_input, tensor_output = tf.import_graph_def(graph_def, name='', return_elements=['input_1:0', 'output_0:0'])
    x = np.random.randn(*tensor_input.shape).astype('float32')
    predictions = sess.run(tensor_output, {tensor_input: x})
    ti = time.time()
    predictions = sess.run(tensor_output, {tensor_input: x})
    print(time.time() - ti)

all_tensors = [tensor for op in tf.get_default_graph().get_operations() for tensor in op.values()]
