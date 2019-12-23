using PyCall: python

run(`$python -m pip install tensorflow-gpu==1.14 keras==2.2.5 keras-tcn`)
run(`$python -m pip install onnx onnxruntime onnxmltools ngraph_tensorflow_bridge`)
run(`$python -m pip install git+https://github.com/OverLordGoldDragon/keras-adamw.git`)
run(`$python -m pip install git+https://github.com/AStupidBear/keras_lr_finder.git`)