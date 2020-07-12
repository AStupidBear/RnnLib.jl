using Random, Statistics, Test
using Parameters, DataStructures
using PyCall, PyCallUtils, PandasLite
using MLSuiteBase, RnnLib

Random.seed!(1234)

@from sklearn.metrics imports accuracy_score, r2_score

function SequentialMNIST(x, y; lag = 300)
    x = reshape(x, :, size(x, 3))' ./ 255f0
    x = hcat(x, zeros(Float32, size(x, 1), lag))
    x = reshape(x, 1, size(x)...)
    y = reshape(y, 1, :)
    return  x, y
end

function CopyFirstInput(n; lag = 300)
    x = randn(Float32, 1, n, lag)
    y = x[:, :, 1]
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

function generate_samples(name)
    if name == "CopyFirstInput"
        x_trn, y_trn = CopyFirstInput(45000; lag = 300)
        x_tst, y_tst = CopyFirstInput(5000; lag = 300)
    elseif name == "Denoising"
        x_trn, y_trn = Denoising(45000; seqlen = 400, lag = 200)
        x_tst, y_tst = Denoising(5000; seqlen = 400, lag = 200)
    elseif name == "SequentialMNIST"
        x_trn, y_trn = SequentialMNIST(MNIST.traindata()...)
        x_tst, y_tst = SequentialMNIST(MNIST.testdata()...)
    end
    return x_trn, y_trn, x_tst, y_tst
end

grid = Any[
    [
        [
            "layer" => ["AHLN", "TCN"],
            "kernel_size" => [3, 5],
            "use_skip_conn" => [true, false],
            "recept_field_ratio" => [0.1, 0.3, 0.5, 0.7],
            "dropout" => [0.0, 0.2],
            "use_batch_norm" => [true, false],
            "lr" => [1e-3, 1e-2],
        ],
        [
            "layer" => ["ResNet", "Inception"],
            "kernel_size" => [3, 5, 7],
            "dropout" => [0.0, 0.2],
            "use_batch_norm" => [true, false],
            "lr" => [1e-3, 1e-2],
        ],
        [
            "layer" => ["GRU", "BRU", "nBRU", "IndRNN"],
            "use_skip_conn" => [true, false],
            "dropout" => [0.0, 0.2],
            "lr" => [1e-3, 1e-4],
        ],
        [
            "layer" => ["Rocket"],
            "recept_field_ratio" => [0.1, 0.3, 0.5, 0.7],
            "lr" => [1e-3, 1e-4, 1e-5],
        ]
    ],
    [
        [
            "dset" => ["CopyFirstInput"],
            "hidden_sizes" => ["100,100"],
            "batch_size" => [100],
            "epochs" => [60],
        ],
        [
            "dset" => ["Denoising"],
            "hidden_sizes" => ["100,100,100,100"],
            "batch_size" => [100],
            "epochs" => [80],
        ],
        [
            "dset" => ["SequentialMNIST"],
            "hidden_sizes" => ["300,300"],
            "batch_size" => [300],
            "epochs" => [20],
        ]
    ],
]
param = @grid 100 gridparams(grid)
param = DefaultDict(nothing, param)
@unpack dset, layer, lr, epochs, dropout, hidden_sizes, use_batch_norm,
        use_skip_conn, kernel_size, recept_field_ratio = param

x_trn, y_trn, x_tst, y_tst = generate_samples(dset)
if dset == "SequentialMNIST"
    loss, out_dim, fscore = "spcce", 10, accuracy_score
else
    loss, out_dim = "mse", 0, r2_score
end
recept_field = ceil(Int, size(x_trn, 3) * recept_field_ratio)

model = RnnModel(
    layer = layer, hidden_sizes = hidden_sizes, kernel_size = kernel_size, recept_field = recept_field, 
    out_dim = out_dim, loss = loss, lr = lr, epochs = epochs, validation_split = 0.1
)
RnnLib.fit!(model, x_trn, y_trn)
ŷ_trn = RnnLib.predict(model, x_trn)
ŷ_tst = RnnLib.predict(model, x_tst)

sr = Series(param)
sr["recept_field"] = recept_field
sr["out_dim"] = out_dim
sr["train_shape"] = size(x_trn)
sr["test_shape"] = size(x_tst)
sr["train_score"] = fscore(vec(y_trn), vec(ŷ_trn))
sr["teest_score"] = fscore(vec(y_tst), vec(ŷ_tst))
csv = joinpath(mkpath(expanduser("~/job/longmem")), "result.csv")
df = isfile(csv) ? pd.read_csv(csv) : DataFrame()
df = df.append(sr, ignore_index = true).dropna().drop_duplicates()
df.to_csv(csv, index = false)