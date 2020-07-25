using PyCall: python

run(`$python -m pip install tensorflow keras==2.3.1 keras-tcn keras-adamw`)
run(`$python -m pip install onnx onnxruntime onnxmltools keras2onnx`)
run(`$python -m pip install alt-model-checkpoint h5py hdf5plugin Cython`)
run(`$python -m pip install "pandas<=0.25.3,!=0.24" "sktime<0.4"`)
llvm_version = Base.libllvm_version
if llvm_version >= v"9.0.0"
    run(`$python -m pip install numba llvmlite==0.33`)
elseif llvm_version >= v"7.0.0"
    run(`$python -m pip install numba llvmlite==0.32`)
elseif llvm_version >= v"6.0.0"
    run(`$python -m pip install numba llvmlite==0.26`)
elseif llvm_version >= v"5.0.0"
    run(`$python -m pip install numba llvmlite==0.22`)
elseif llvm_version >= v"4.0.0"
    run(`$python -m pip install numba llvmlite==0.20`)
end