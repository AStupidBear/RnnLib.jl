module RnnLib

using Distributed, Random, Statistics, Parameters
using PyCall, HDF5, PyCallUtils, HDF5Utils
import ScikitLearnBase: BaseEstimator, fit!, predict, predict_proba, is_classifier
import MLSuite: reset!, modelhash, signone

export RnnModel, RnnClassifier, RnnRegressor

include("util.jl")
include("rnn.jl")

end # module