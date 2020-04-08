
# convert model
keras2tf = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keras2tf.py')
os.system('CUDA_VISIBLE_DEVICES=-1 python %s --input_model=rnn.h5 --output_model=rnn.pb --output_nodes_prefix=output_ 2> /dev/null' % keras2tf)
os.system('CUDA_VISIBLE_DEVICES=-1 python -m tensorflow.python.tools.optimize_for_inference --input=rnn.pb --output=rnn_opt.pb --input_names=input_1 --output_names=output_1 2> /dev/null')
if layer in ('Rocket', 'IndRNN'):
    exit()
if i.shape[1] is not None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    open("rnn.tflite","wb").write(converter.convert())
onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, 'rnn.onnx')

# inference
import onnxruntime as ort
so = ort.SessionOptions()
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
sess = ort.InferenceSession("rnn.onnx", so)
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
if input_shape[1] is None:
    input_shape[1] = 666
img = np.random.randn(32, *input_shape[1:]).astype('float32')
for provider in sess.get_providers():
    sess.set_providers([provider])
    sess.run(None, {input_name: img})[0]
    ti = time.time()
    sess.run(None, {input_name: img})[0]
    print('onnx-' + provider, 'time: ', time.time() - ti)
model = load_model('rnn.h5', compile=False, custom_objects=custom_objects)
model(img, training=False)
ti = time.time()
model(img, training=False)
print('keras time: ', time.time() - ti)
ti = time.time()
model.predict(img, batch_size=32)
print('keras time: ', time.time() - ti)
if os.path.isfile('rnn.tflite'):
    interpreter = tf.lite.Interpreter("rnn.tflite")
    interpreter.allocate_tensors()
    input_tensor = interpreter.tensor(interpreter.get_input_details()[0]["index"])
    output_tensor = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    interpreter.invoke()
    ti = time.time()
    interpreter.invoke()
    print('tflite time: ', time.time() - ti)
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    with gfile.GFile('rnn_opt.pb','rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tensor_input, tensor_output = tf.import_graph_def(graph_def, return_elements=['input_1:0', 'output_1:0'])
    sess.run(tensor_output, {tensor_input: img})
    ti = time.time()
    sess.run(tensor_output, {tensor_input: img})
    print('tf time: ', time.time() - ti)

# # ngraph inference
# import onnx
# import ngraph as ng
# from ngraph_onnx.onnx_importer.importer import import_onnx_model
# ng_func = import_onnx_model(onnx.load('rnn.onnx'))
# ngrt = ng.runtime(backend_name='CPU')
# ng_comp = ngrt.computation(ng_func)
# img = np.random.randn(32, 666, 30).astype('float32')
# ng_comp(img)

# # caffe2 inference
# from caffe2.python.onnx.backend import run_model
# img = np.random.randn(32, 666, 30).astype(np.float32)
# model = onnx.load('rnn.onnx')
# outputs = run_model(model, [img])
