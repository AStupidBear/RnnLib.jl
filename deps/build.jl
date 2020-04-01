using PyCall: python

run(`$python -m pip install tensorflow-gpu==1.14 keras==2.2.5 keras-tcn==2.8.3`)
run(`$python -m pip install onnx onnxruntime onnxmltools ngraph_tensorflow_bridge`)
run(`$python -m pip install keras-layer-normalization`)
run(`$python -m pip install git+https://github.com/AStupidBear/keras-adamw.git`)
run(`$python -m pip install git+https://github.com/AStupidBear/keras_lr_finder.git`)
if get(ENV, "USE_MKL", "0") == "1"
    run(`$python -m pip install pip install https://sourceforge.net/projects/bearapps/files/onnxruntime-1.0.0-cp36-cp36m-linux_x86_64.whl`)
    run(`$python -m pip install pip install https://github.com/evdcush/TensorFlow-wheels/releases/download/tf-1.12.0-gpu-10.0/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl`)
end