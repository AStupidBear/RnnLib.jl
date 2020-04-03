using PyCall: python

run(`$python -m pip install tensorflow keras`)
run(`$python -m pip install keras-layer-normalization phased_lstm_keras`)
run(`$python -m pip install git+https://github.com/philipperemy/keras-tcn.git`)
run(`$python -m pip install git+https://github.com/OverLordGoldDragon/keras-adamw.git`)
run(`$python -m pip install onnx onnxruntime onnxmltools ngraph_tensorflow_bridge`)
run(`$python -m pip install git+https://github.com/microsoft/onnxconverter-common.git`)
run(`$python -m pip install git+https://github.com/onnx/keras-onnx.git`)
if get(ENV, "USE_MKL", "0") == "1"
    run(`$python -m pip install https://sourceforge.net/projects/bearapps/files/onnxruntime_dnnl-1.2.0-cp36-cp36m-linux_x86_64.whl`)
end