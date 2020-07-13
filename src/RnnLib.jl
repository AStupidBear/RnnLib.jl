module RnnLib

using Random, Parameters
using HDF5, HDF5Utils, MLSuiteBase
import ScikitLearnBase: BaseEstimator, fit!, predict, predict_proba, is_classifier
import MLSuiteBase: reset!, modelhash

export RnnModel, RnnClassifier, RnnRegressor

include("rnn.jl")

end # module