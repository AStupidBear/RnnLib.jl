using PyCall: python

run(`$python -m pip install tensorflow==2.1.0 keras==2.3.1 keras-tcn keras-adamw alt-model-checkpoint h5py hdf5plugin`)
run(`$python -m pip install --no-cache-dir horovod`)
run(`$python -m pip install onnx onnxruntime onnxmltools keras2onnx`)
run(`$python -m pip install numba==0.41 llvmlite==0.26`)