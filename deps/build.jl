using PyCall: python

run(`$python -m pip install keras onnx onnxruntime onnxmltool ngraph_tensorflow_bridge`)
run(`$python -m pip install git+https://github.com/OverLordGoldDragon/keras-adamw.git`)
run(`$python -m pip install git+https://github.com/AStupidBear/keras_lr_finder.git`)