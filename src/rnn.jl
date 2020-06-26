mutable struct RnnModel <: BaseEstimator
    rnn::Vector{UInt8}
    config::Dict{Symbol, String}
end

is_classifier(m::RnnModel) = occursin(r"ce|crossentropy", m.config["loss"])

const rnnpy = joinpath(@__DIR__, "rnn.py")

RnnModel(;ka...) = RnnModel(UInt8[], Dict(k => string(v) for (k, v) in ka))
RnnRegressor(;ka...) = RnnModel(;loss = "mse", ka...)
RnnClassifier(;ka...) = RnnModel(;loss = "bce", ka...)

concat_config(config) = reduce(vcat, [v == "true" ? ["--$k"] : v == "false" ? [] : ["--$k", v] for (k, v) in config])

function fit!(m::RnnModel, h5::String)
    @unpack rnn, config = m
    args = concat_config(config)
    cmd = `python $rnnpy --data_path $h5 $args`
    println(cmd); run(cmd)
    m.rnn = read("model.h5")
    return m
end

function predict(m::RnnModel, h5::String)
    @unpack rnn, config = m
    !isempty(rnn) && write("model.h5", rnn)
    args = concat_config(config)
    h5p = joinpath(dirname(h5), randstring())
    run(`python $rnnpy --data_path $h5 --pred_path $h5p --test $args`)
    return h5p
end

function fit!(m::RnnModel, x, y, w = nothing; columns = nothing)
    if m.config["loss"] == "bce" && minimum(y) < 0
        y = (y .+ 1) ./ 2
    end
    fit!(m, dump_rnn_data(x, y, w))
end

function predict_proba(m::RnnModel, x)
    ŷ = h5read(predict(m, dump_rnn_data(x)), "p")
    size(ŷ, 1) > 1 ? ŷ : vcat(1 .- ŷ,  ŷ)
end

function predict(m::RnnModel, x)
    ŷ = h5read(predict(m, dump_rnn_data(x)), "p")
    !is_classifier(m) && return ŷ
    if size(ŷ, 1) == 1
        signone.(ŷ .- 0.5)
    else
        [I[1] - 1 for I in argmax(ŷ, dims = 1)]
    end
end

reset!(m::RnnModel) = empty!(m.rnn)

modelhash(m::RnnModel) = hash(m.rnn)

function dump_rnn_data(x, y = nothing, w = nothing)
    h5 = abspath(randstring() * ".rnn")
    h5open(h5, "w") do fid
        write_batch(fid, "x", x)
        isnothing(y) || write_batch(fid, "y", y)
        isnothing(w) || write_batch(fid, "w", w)
    end
    atexit(() -> rm(h5, force = true))
    return h5
end