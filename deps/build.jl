using PyCall: python

run(`$python -m pip install keras onnx onnxruntime`)
run(`$python -m pip install git+https://github.com/OverLordGoldDragon/keras-adamw.git`)