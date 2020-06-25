using PyCall: python

run(`$python -m pip install tensorflow keras onnx onnxruntime alt-model-checkpoint h5py hdf5plugin`)
run(`$python -m pip install --no-cache-dir horovod`)
run(`$python -m pip install https://github.com/philipperemy/keras-tcn/archive/master.zip`)
run(`$python -m pip install https://github.com/OverLordGoldDragon/keras-adamw/archive/master.zip`)
run(`$python -m pip install https://github.com/microsoft/onnxconverter-common/archive/master.zip`)
run(`$python -m pip install https://github.com/onnx/onnxmltools/archive/master.zip`)
run(`$python -m pip install https://github.com/onnx/keras-onnx/archive/master.zip`)
run(`$python -m pip install numba==0.41 llvmlite==0.26`)