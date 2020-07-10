using RnnLib
using MLSuiteBase
using Random
using Statistics
using Test

cd(mktempdir())

F, N, T = 3, 32, 100
Random.seed!(1234)
x = randn(Float32, F, N, T)
y = mean(x, dims = 1) * √F
w = ones(Float32, N, T)

layer, binary = "Conv", true
for layer in ["Conv", "ResNet", "Inception", "TCN", "Rocket", "LSTM", "GRU", "IndRNN"]
    model = RnnRegressor(layer = layer, hidden_sizes = "10", epochs = 200, validation_split = 0)
    RnnLib.fit!(model, x, y, w)
    ŷ = RnnLib.predict(model, x)
    res = mean(abs2, vec(y) .- vec(ŷ))
    @test res < 0.1
    for binary in [true, false]
        y′ = binary ? signone.(y) : @. ifelse(abs(y) > 0.5f0, sign(y) + 1f0, 1f0)
        model = RnnClassifier(
            layer = layer, hidden_sizes = "10", epochs = 200, validation_split = 0,
            loss = binary ? "bce" : "spcce", out_dim = binary ? 1 : 3, lr = 1e-2
        )
        RnnLib.fit!(model, x, y′, w)
        ŷ = RnnLib.predict(model, x)
        prob = RnnLib.predict_proba(model, x)
        res = mean(abs, vec(y′) .- vec(ŷ))
        @test res < 0.3
    end
end

# model = RnnClassifier(layer = "Rocket", loss = "direct", hidden_sizes = "5", epochs = 300, validation_split = 0)
# RnnLib.fit!(model, x, y, w)