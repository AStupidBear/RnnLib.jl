module RnnLib

using Distributed, Random, Statistics, Parameters
using PyCall, HDF5, PyCallUtils, HDF5Utils, MLSuiteBase
import ScikitLearnBase: BaseEstimator, fit!, predict, predict_proba, is_classifier
import MLSuiteBase: reset!, modelhash

export RnnModel, RnnClassifier, RnnRegressor

include("util.jl")
include("rnn.jl")

end # module