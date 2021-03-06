using Random, Statistics, Test
using MLSuiteBase, RnnLib

cd(mktempdir())

Random.seed!(1234)

F, N, T = 3, 32, 100
x = randn(Float32, F, N, T)
y = mean(x, dims = 1) * sqrt(1f0 * F)
w = ones(Float32, N, T)

for layer in ["Conv", "AHLN", "ResNet", "Inception", "TCN", "Rocket", "GRU", "BRU", "nBRU", "IndRNN"]
    layer == "Rocket" && continue
    model = RnnRegressor(layer = layer, hidden_sizes = "10", lr = 1e-2, epochs = 200, patience = 10)
    RnnLib.fit!(model, x, y, w)
    ŷ = RnnLib.predict(model, x)
    mse = mean(abs2, vec(y) .- vec(ŷ))
    layer != "Rocket" && @test mse < 0.1
    for binary in [true, false]
        y′ = binary ? (@. ifelse(y > 0f0, 1f0, 0f0)) : (@. ifelse(abs(y) > 0.5f0, sign(y) + 1f0, 1f0))
        model = RnnClassifier(
            layer = layer, hidden_sizes = "10", output_dim = binary ? 1 : 3, 
            loss = binary ? "bce" : "spcce", lr = 1e-2, epochs = 200, patience = 10
        )
        RnnLib.fit!(model, x, y′, w)
        ŷ = RnnLib.predict(model, x)
        prob = RnnLib.predict_proba(model, x)
        acc = mean(isapprox.(vec(y′), vec(ŷ)))
        layer != "Rocket" && @test acc > 0.9
    end
end

model = RnnClassifier(layer = "Rocket", loss = "direct", hidden_sizes = "10", epochs = 100, validation_split = 0)
RnnLib.fit!(model, x, y, w)
ŷ = RnnLib.predict(model, x)[1:1, :, :]
@test cor(vec(ŷ), vec(y)) > 0.5