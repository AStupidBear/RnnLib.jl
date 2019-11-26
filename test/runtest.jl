using RnnLib
using MLSuite
using Statistics
using Test

cd(mktempdir())

F, N, T = 10, 100, 10
x = randn(Float32, F, N, T)
w = rand(Float32, N, T)
y = mean(x, dims = 1)

model = RnnRegressor(epochs = 30)
RnnLib.fit!(model, x, y, w)
ŷ = RnnLib.predict(model, x)
r2 = r2_score(vec(ŷ), vec(y))
@test r2 > 0.5

model = RnnClassifier(loss = binary ? "bce" : "spcce")
for binary in [true, false][2:2]
    y′ = binary ? signone.(y) : @. ifelse(abs(y) > 0.1, sign(y) + 1.0, 1.0)
    RnnLib.fit!(model, x, y′, w)
    ŷ = RnnLib.predict(model, x)
    prob = RnnLib.predict_proba(model, x)
    acc = accuracy_score(vec(ŷ), vec(y′))
    @assert acc > 0.6
end