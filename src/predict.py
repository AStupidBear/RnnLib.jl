import onnx
import onnxruntime as rt

model = onnx.load("rnn.onnx")
siz = 0
for tensor in model.graph.initializer:
    data = tensor.float_data
    siz += length(data)
    for i in range(len(data)):
        data[i] = data[i]
onnx.save_model(model, "rnn.onnx")

sess = rt.InferenceSession("abc.onnx")
sess = rt.InferenceSession("rnn.onnx")
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
y = sess.run(None, {input_name: x})[0]