using Random, Statistics, Test
using Parameters, DataStructures
using PyCall, PyCallUtils, PandasLite
using MLSuiteBase, RnnLib

Random.seed!(1234)

@from sklearn.metrics imports accuracy_score, r2_score
@from tensorflow.keras.datasets imports imdb, mnist
@from tensorflow.keras.preprocessing imports sequence

function CopyFirstInput(n; seqlen = 300)
    x = randn(Float32, 1, n, seqlen)
    return x, x[:, :, 1]
end

function AddingProblem(n; seqlen = 600)
    x_num = rand(Float32, 1, n, seqlen)
    x_mask = zeros(Float32, 1, n, seqlen)
    y = zeros(Float32, 1, n)
    for i in 1:n
        ts = randperm(seqlen)[1:2]
        x_mask[1, i, ts] .= 1
        y[1, i] = sum(x_num[1, i, ts])
    end
    return vcat(x_num, x_mask), y
end

function CopyMemory(n; seqlen = 601, memlen = 10)
    seq = rand(1f0:8f0, (n, memlen))
    zero = fill(0f0, (n, seqlen))
    marker = fill(9f0, (n, memlen + 1))
    zeroy = fill(0f0, (n, memlen + seqlen + 1))
    x = hcat(seq, zero, marker)
    y = hcat(zeroy, seq)
    x = reshape(x, 1, size(x)...)
    y = reshape(y, 1, size(y)...)
    return x, y
end

function Denoising(n; seqlen = 400, lag = 300)
    x = randn(Float32, 2, n, seqlen)
    x[2, :, :] .= -1
    x[:, :, end] .= [0, 1]
    ts = randperm(seqlen - lag)[1:5]
    x[2, :, ts] .= 0
    y = permutedims(x[1, :, ts])
    return x, y
end

function SequentialMNIST(x, y; lag = 300)
    x = reshape(x, size(x, 1), :) ./ 255f0
    x = hcat(x, zeros(Float32, size(x, 1), lag))
    x = reshape(x, 1, size(x)...)
    y = Int32.(reshape(y, 1, :))
    return  x, y
end

function IMDB(x, y; maxlen = 500)
    x = sequence.pad_sequences(x, maxlen)
    y = Int32.(reshape(y, 1, :))
    return  x, y
end

function generate_samples(name)
    if name == "CopyFirstInput"
        x_trn, y_trn = CopyFirstInput(45000; seqlen = 300)
        x_tst, y_tst = CopyFirstInput(5000; seqlen = 300)
    elseif name == "AddingProblem"
        x_trn, y_trn = AddingProblem(200000; seqlen = 600)
        x_tst, y_tst = AddingProblem(40000; seqlen = 600)
    elseif name == "CopyMemory"
        x_trn, y_trn = CopyMemory(30000; seqlen = 601, memlen = 10)
        x_tst, y_tst = CopyMemory(6000; seqlen = 601, memlen = 10)
    elseif name == "Denoising"
        x_trn, y_trn = Denoising(45000; seqlen = 400, lag = 200)
        x_tst, y_tst = Denoising(5000; seqlen = 400, lag = 200)
    elseif name == "SequentialMNIST"
        (x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()
        x_trn, y_trn = SequentialMNIST(x_trn, y_trn)
        x_tst, y_tst = SequentialMNIST(x_tst, y_tst)
    elseif name == "IMDB"
        (x_trn, y_trn), (x_tst, y_tst) = imdb.load_data(num_words = 5000)
        x_trn, y_trn = IMDB(x_trn, y_trn, maxlen = 500)
        x_tst, y_tst = IMDB(x_tst, y_tst, maxlen = 500)
    end
    return x_trn, y_trn, x_tst, y_tst
end

grid = [
    "dset" => ["CopyFirstInput", "AddingProblem", "CopyMemory", "Denoising", "SequentialMNIST", "IMDB"][end:end],
    "hidden_sizes" => ["32,32", "64,64", "128,128"],
    "l2" => [0, 1e-5, 1e-4],
    "batch_size" => [128],
    "epochs" => [60],
    [
        [
            "layer" => ["AHLN", "TCN"],
            "hidden_sizes" => ["32,32"],
            "kernel_size" => [3, 5],
            "use_skip_conn" => [true, false],
            "recept_field_ratio" => [0.1, 0.3, 0.5, 0.7],
            "dropout" => [0.0, 0.1],
            "use_batch_norm" => [true, false],
            "lr" => [1e-3, 1e-2],
        ],
        [
            "layer" => ["ResNet", "Inception"],
            "kernel_size" => [3, 5, 7],
            "dropout" => [0.0, 0.1],
            "use_batch_norm" => [true, false],
            "lr" => [1e-3, 1e-2],
        ],
        [
            "layer" => ["GRU", "BRU", "nBRU", "IndRNN"],
            "use_skip_conn" => [true, false],
            "dropout" => [0.0, 0.1],
            "use_batch_norm" => [true, false],
            "lr" => [1e-3, 1e-4],
        ],
        [
            "layer" => ["Rocket"],
            "recept_field_ratio" => [0.1, 0.3, 0.5, 0.7],
            "lr" => [1e-3, 1e-4, 1e-5],
        ]
    ],
]
param = @grid 100 gridparams(grid)
param = DefaultDict(nothing, param)
@unpack dset, layer, lr, epochs, dropout, hidden_sizes, use_batch_norm,
        use_skip_conn, kernel_size, recept_field_ratio = param

x_trn, y_trn, x_tst, y_tst = generate_samples(dset)
if dset == "SequentialMNIST"
    loss, output_dim, fscore = "spcce", 10, accuracy_score
elseif dset == "IMDB"
    loss, output_dim, fscore = "bce", 0, accuracy_score
else
    loss, output_dim, fscore = "mse", 0, r2_score
end
recept_field = ceil(Int, size(x_trn, 3) * something(recept_field_ratio, 0))

model = RnnModel(
    layer = layer, hidden_sizes = hidden_sizes, kernel_size = kernel_size, recept_field = recept_field, 
    output_dim = output_dim, loss = loss, lr = lr, epochs = epochs, patience = 3, validation_split = 0.1,
)
RnnLib.fit!(model, x_trn, y_trn)
ŷ_trn = RnnLib.predict(model, x_trn)
ŷ_tst = RnnLib.predict(model, x_tst)

sr = Series(param)
sr["recept_field"] = recept_field
sr["output_dim"] = output_dim
sr["train_shape"] = size(x_trn)
sr["test_shape"] = size(x_tst)
sr["train_score"] = fscore(vec(y_trn), vec(ŷ_trn))
sr["teest_score"] = fscore(vec(y_tst), vec(ŷ_tst))
csv = joinpath(mkpath(expanduser("~/job/longmem")), "result.csv")
df = isfile(csv) ? pd.read_csv(csv) : DataFrame()
df = df.append(sr, ignore_index = true).dropna().drop_duplicates()
df.to_csv(csv, index = false)