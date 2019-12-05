@with_kw mutable struct RnnModel <: BaseEstimator
    rnn::Vector{UInt8} = UInt8[]
    seq_size::Int = 0
    n_jobs::Int = 1
    warm_start::Int = 0
    lr::Float32 = 1f-3
    batch_size::Int = 32
    epochs::Int = 10
    layer::String = "TCN"
    out_activation::String = "linear"
    hidden_sizes::String = "10,10"
    loss::String = "mse"
    kernel_size::Int = 2
    dilations::String = "1,2,4,8,16,32,64"
    l2::Float32 = 0f-4
    dropout_rate::Float32 = 0
    use_batch_norm::Int = 0
    commission::Float32 = 2f-4
    out_seq::Bool = true
    out_dim::Int = 0
    validation_split::Float32 = 0.3
    patience::Int = 10
end

is_classifier(m::RnnModel) = occursin(r"ce|crossentropy", m.loss)

const rnnpy = joinpath(@__DIR__, "rnn.py")

function fit!(m::RnnModel, x, y, w = nothing; columns = nothing)
    columns = something(columns, string.(1:size(x, 1)))
    @unpack rnn, seq_size, n_jobs, warm_start, lr = m
    @unpack batch_size, epochs, layer, out_activation = m
    @unpack hidden_sizes, loss, kernel_size, dilations = m
    @unpack l2, dropout_rate, use_batch_norm, commission = m
    @unpack validation_split, patience = m
    out_seq, out_dim = ndims(y) == 3, size(y, 1)
    pnl_scale = Meta.parse(get(ENV, "PNL_SCALE", "1"))
    loss == "bce" && out_dim > 1 && (loss = "cce")
    loss == "bce" && out_dim == 1 && length(unique(y)) > 2 && (loss = "spcce")
    if loss == "bce" && minimum(y) < 0
        y = (y .+ 1f0) ./ 2f0
    elseif loss == "spcce"
        out_dim = Int(maximum(y)) + 1
    end
    if occursin(r"LSTM|GRU", layer)
        shuffle, N = false, size(x, 2)
        batch_size = max(1, batch_size ÷ N) * N
        @pack! m = batch_size
    else
        shuffle = true
    end
    hosts = join(pmap(n -> gethostname(), 1:max(n_jobs, nworkers())), ',')
    dst = dump_rnn_data(x, y, w, out_dim = out_dim, seq_size = seq_size, shuffle = shuffle)
    !isempty(rnn) && write("rnn.h5", rnn)
    run(`mpirun --host $hosts python $rnnpy --data $dst --file rnn.h5
        --warm_start $warm_start --lr $lr --batch_size $batch_size --epochs $epochs
        --layer $layer --out_activation $out_activation --hidden_sizes $hidden_sizes
        --loss $loss --kernel_size $kernel_size --dilations $dilations
        --l2 $l2 --dropout_rate $dropout_rate --use_batch_norm $use_batch_norm
        --commission $commission --pnl_scale $pnl_scale --out_dim $out_dim
        --validation_split $validation_split --patience $patience`)
    m.rnn = read("rnn.h5")
    cp("rnn.h5", "rnn.h5.bak", force = true)
    @pack! m = out_dim, out_seq
    warm_start == 0 && rm("rnn.h5")
    return m
end

function predict_rnn(m::RnnModel, x)
    @unpack rnn, batch_size, out_seq, out_dim = m
    dst = dump_rnn_data(x, out_seq = out_seq, out_dim = out_dim)
    write("rnn.h5", rnn)
    run(`python $rnnpy --test 1 --data $dst
        --file rnn.h5 --batch_size $batch_size`)
    ŷ = h5read(dst, "p")
    perm = (1, ndims(ŷ):-1:2...)
    ŷ = permutedims(ŷ, perm)
    rm("rnn.h5")
    return ŷ
end

function predict_proba(m::RnnModel, x)
    ŷ = predict_rnn(m, x)
    size(ŷ, 1) > 1 ? ŷ :
    cat(1 .- ŷ,  ŷ, dims = 1)
end

function predict(m::RnnModel, x)
    ŷ = predict_rnn(m, x)
    !is_classifier(m) && return ŷ
    if size(ŷ, 1) == 1
        signone.(ŷ .- 0.5)
    else
        [I[1] - 1 for I in argmax(ŷ, dims = 1)]
    end
end

reset!(m::RnnModel) = empty!(m.rnn)

function consistent(dst, x)
    !isfile(dst) && return false
    h5open(dst, "r") do fid
        size(fid["x"]) == size(x) &&
        isapprox(x[:, :, 1], fid["x"][:, :, 1])
    end
end

function dump_rnn_data(x, y = nothing, w = nothing; out_dim = 1, out_seq = true, seq_size = size(x, 3), shuffle = false)
    seq_size = seq_size < 1 ? size(x, 3) : seq_size
    rng = MersenneTwister(hash(Main))
    dst = abspath(randstring(rng) * ".rnn")
    if isnothing(y)
        y = fill(0f0, out_dim, (out_seq ? size(x)[2:end] : size(x, 2))...)
    end
    if isnothing(w)
        w = fill(1f0, 1, size(y)[2:end]...)
    else
        w = reshape(w, 1, size(y)[2:end]...)
    end
    ŷ = fill(0f0, out_dim, size(y)[2:end]...)
    if size(x, 3) / seq_size < 5 && ndims(y) == 3
        xᵇ = batchfirst(rebatchseq(x, seq_size))
        yᵇ = batchfirst(rebatchseq(y, seq_size))
        ŷᵇ = batchfirst(rebatchseq(y, seq_size))
        wᵇ = batchfirst(rebatchseq(w, seq_size))
    else
        xᵇ, yᵇ = batchfirst(x), batchfirst(y)
        ŷᵇ, wᵇ = batchfirst(ŷ), batchfirst(w)
    end
    if shuffle
        rng = MersenneTwister(1234)
        is = randperm(rng, size(xᵇ, 3))
        xᵇ, yᵇ = cview(xᵇ, is), cview(yᵇ, is)
        ŷᵇ, wᵇ = cview(ŷᵇ, is), cview(wᵇ, is)
    end
    wᵇ = reshape(wᵇ, size(yᵇ)[2:end]...)
    if !isfile(dst) || !consistent(dst, xᵇ)
        h5save(dst, (x = xᵇ, y = yᵇ, w = wᵇ, p = ŷᵇ))
    else
        h5open(dst, "r+") do fid
            write_batch(fid, "y", yᵇ)
            write_batch(fid, "w", wᵇ)
            write_batch(fid, "p", ŷᵇ)
        end
    end
    atexit(() -> rm(dst, force = true))
    return dst
end

modelhash(m::RnnModel) = hash(m.rnn)

function receptive_field(m::RnnModel)
    @unpack kernel_size, dilations, hidden_sizes = m
    dilations = Meta.parse.(split(dilations, ','))
    hidden_sizes = Meta.parse.(split(hidden_sizes, ','))
    dilations[end] * length(hidden_sizes) * kernel_size
end

receptive_field(m) = 1

RnnRegressor(;ka...) = RnnModel(;out_activation = "tanh", warm_start = 1, ka...)

RnnClassifier(;ka...) = RnnModel(;loss = "bce", ka...)