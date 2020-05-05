virtualenv ${JULIA_DEPOT_PATH-~/.julia}/../.virtualenv/horovod
pip install https://github.com/AStupidBear/tensorflow-mkl/releases/download/2.1.0/tensorflow-compat-2.1.0-cp36-cp36m-linux_x86_64.whl
pip install https://github.com/AStupidBear/onnxruntime-dnnl/releases/download/1.2/onnxruntime_dnnl-1.2.0-cp36-cp36m-linux_x86_64.whl
pip install keras onnx alt-model-checkpoint h5py hdf5plugin
pip install --no-cache-dir horovod
pip install https://github.com/philipperemy/keras-tcn/archive/master.zip
pip install https://github.com/OverLordGoldDragon/keras-adamw/archive/master.zip
pip install https://github.com/microsoft/onnxconverter-common/archive/master.zip
pip install https://github.com/onnx/onnxmltools/archive/master.zip
pip install https://github.com/onnx/keras-onnx/archive/master.zip
pip install numba==0.41 llvmlite==0.26 numpy==1.17.0