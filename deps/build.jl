using PyCall: python

run(`$python -m pip install https://sourceforge.net/projects/bearapps/files/tensorflow-2.1.0-cp36-cp36m-linux_x86_64.whl`)
run(`$python -m pip install https://sourceforge.net/projects/bearapps/files/onnxruntime_dnnl-1.2.0-cp36-cp36m-linux_x86_64.whl`)
run(`$python -m pip install keras onnx alt-model-checkpoint h5py hdf5plugin`)
run(`$python -m pip install --no-cache-dir horovod`)
run(`$python -m pip install https://github.com/philipperemy/keras-tcn/archive/master.zip`)
run(`$python -m pip install https://github.com/OverLordGoldDragon/keras-adamw/archive/master.zip`)
run(`$python -m pip install https://github.com/microsoft/onnxconverter-common/archive/master.zip`)
run(`$python -m pip install https://github.com/onnx/onnxmltools/archive/master.zip`)
run(`$python -m pip install https://github.com/onnx/keras-onnx/archive/master.zip`)
run(`$python -m pip install numba==0.41 llvmlite==0.26 numpy==1.17.0`)