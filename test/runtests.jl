using RnnLib
using MLSuiteBase
using Random
using Statistics
using Test

ENV["USE_GPU"] = 0

cd(mktempdir())

F, N, T = 3, 200, 100
Random.seed!(1234)
x = rand(UInt8, F, N, T)
y = mean(x, dims = 1)
y = mean(x ./ 128 .- 1, dims = 1)
w = rand(Float32, N, T)

for layer in ["ResNet", "Inception", "TCN", "Rocket"]
    model = RnnRegressor(layer = layer, hidden_sizes = "5", epochs = 200, validation_split = 0)
    RnnLib.fit!(model, x, y, w)
    ŷ = RnnLib.predict(model, x)
    res = mean(abs, vec(y) .- vec(ŷ))
    @test res < 0.3

    model = RnnClassifier(layer = layer, hidden_sizes = "5", epochs = 200, validation_split = 0)
    for binary in [true, false]
        y′ = binary ? signone.(y) : @. ifelse(abs(y) > 0.1, sign(y) + 1.0, 1.0)
        RnnLib.fit!(model, x, y′, w)
        ŷ = RnnLib.predict(model, x)
        prob = RnnLib.predict_proba(model, x)
        res = mean(abs, vec(y′) .- vec(ŷ))
        @test res < 0.3
    end
end

model = RnnClassifier(layer = "ResNet", loss = "direct", hidden_sizes = "5", epochs = 200)
RnnLib.fit!(model, x, y, w)