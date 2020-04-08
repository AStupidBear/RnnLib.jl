@with_kw mutable struct RnnModel <: BaseEstimator
    rnn::Vector{UInt8} = UInt8[]
    n_jobs::Int = 1
    warm_start::Int = 0
    optimizer::String = "AdamW"
    lr::Float32 = 1f-3
    sequence_size::Int = 0
    batch_size::Int = 32
    epochs::Int = 100
    layer::String = "AHLN"
    out_activation::String = "linear"
    hidden_sizes::String = "128"
    loss::String = "mse"
    kernel_size::Int = 3
    kernel_sizes::String = "7,9,11"
    pool_size::Int = 1
    max_dilation::Int = 64
    l2::Float32 = 0f-4
    dropout::Float32 = 0
    use_batch_norm::Int = 0
    use_skip_conn::Int = 0
    bottleneck_size::Int = 32
    commission::Float32 = 2f-4
    out_seq::Bool = true
    out_dim::Int = 0
    validation_split::Float32 = 0.2
    patience::Int = 10
    close_thresh::Float32 = 0.5
    eta::Float32 = 0.1
end

is_classifier(m::RnnModel) = occursin(r"ce|crossentropy", m.loss)

const rnnpy = joinpath(@__DIR__, "rnn.py")

function fit!(m::RnnModel, x, y, w = nothing; columns = nothing, pnl_scale = 1)
    columns = something(columns, string.(1:size(x, 1)))
    @unpack rnn, n_jobs, warm_start, optimizer, lr, sequence_size = m
    @unpack batch_size, epochs, layer, out_activation, hidden_sizes = m
    @unpack loss, kernel_size, kernel_sizes, pool_size, max_dilation = m
    @unpack l2, dropout, use_batch_norm, use_skip_conn, bottleneck_size = m
    @unpack commission, validation_split, patience, close_thresh, eta = m
    out_seq, out_dim = ndims(y) == 3, size(y, 1)
    loss == "bce" && out_dim > 1 && (loss = "cce")
    loss == "bce" && out_dim == 1 && length(unique(y)) > 2 && (loss = "spcce")
    if loss == "bce" && minimum(y) < 0
        y = (y .+ 1f0) ./ 2f0
    elseif loss == "spcce"
        out_dim = Int(maximum(y)) + 1
    end
    if occursin(r"LSTM|GRU", layer)
        N = size(x, 2)
        batch_size = max(1, batch_size ÷ N) * N
        @pack! m = batch_size
    end
    hosts = join(pmap(n -> gethostname(), 1:max(n_jobs, nworkers())), ',')
    dst = dump_rnn_data(x, y, w, out_dim, out_seq)
    !isempty(rnn) && write("rnn.h5", rnn)
    if isnothing(Sys.which("mpiexec"))
        exe = `python`
    else
        exe = `mpirun --host $hosts python`
    end
    run(`$exe $rnnpy --data $dst --file rnn.h5 --warm_start $warm_start --optimizer $optimizer
        --lr $lr --batch_size $batch_size --sequence_size $sequence_size --epochs $epochs
        --layer $layer --out_activation $out_activation --hidden_sizes $hidden_sizes
        --loss $loss --kernel_size $kernel_size --kernel_sizes $kernel_sizes --pool_size $pool_size
        --max_dilation $max_dilation --l2 $l2 --dropout $dropout --use_batch_norm $use_batch_norm
        --use_skip_conn $use_skip_conn --bottleneck_size $bottleneck_size --commission $commission
        --pnl_scale $pnl_scale --out_dim $out_dim --validation_split $validation_split 
        --patience $patience --close_thresh $close_thresh --eta $eta`)
    m.rnn = read("rnn.h5")
    @pack! m = out_dim, out_seq
    cp("rnn.h5", "rnn.h5.bak", force = true)
    warm_start == 0 && rm("rnn.h5")
    return m
end

function predict_rnn(m::RnnModel, x)
    @unpack rnn, batch_size, out_seq, out_dim, sequence_size = m
    dst = dump_rnn_data(x, nothing, nothing, out_dim, out_seq)
    write("rnn.h5", rnn)
    run(`python $rnnpy --test 1 --data $dst --file rnn.h5
        --batch_size $batch_size --sequence_size $sequence_size`)
    ŷ = h5read(dst, "p")
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

function dump_rnn_data(x, y, w, out_dim, out_seq)
    rng = MersenneTwister(hash(Main))
    dst = abspath(randstring(rng) * ".rnn")
    if isnothing(y)
        dims = out_seq ? size(x)[2:end] : size(x, 2)
        y = fill(0f0, out_dim, dims...)
    end
    if isnothing(w)
        w = fill(1f0, size(y)[2:end]...)
    else
        w = reshape(w, size(y)[2:end]...)
    end
    ŷ = fill(0f0, out_dim, size(y)[2:end]...)
    if !isfile(dst) || !consistent(dst, x)
        h5save(dst, (x = x, y = y, w = w, p = ŷ))
    else
        h5open(dst, "r+") do fid
            write_batch(fid, "y", y)
            write_batch(fid, "w", w)
            write_batch(fid, "p", ŷ)
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

RnnRegressor(;ka...) = RnnModel(;loss = "mse", ka...)

RnnClassifier(;ka...) = RnnModel(;loss = "bce", ka...)