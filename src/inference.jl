using PyCall, GCMAES, MPI

# isinteractive() && MPI.Init()

pushfirst!(PyVector(pyimport("sys")."path"), @__DIR__)

predict = pyimport("inference").get_predict_function("rnn")

@time predict(rand(Float32, 2, 30000, 30))
# GCMAES.minimize(loss, x0, 0.5, maxfevals = 300000)

# MPI.Finalize()-