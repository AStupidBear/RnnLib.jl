using Random, Statistics, Printf, Test
using Parameters, Glob, ProgressMeter
using PyCall, HDF5, PyCallUtils, HDF5Utils
using PandasLite, MLSuiteBase, RnnLib

@from sktime.utils.load_data imports load_from_tsfile_to_dataframe
@from scipy.interpolate imports interp1d
@from sklearn.preprocessing imports LabelEncoder
@from sklearn.metrics imports accuracy_score

function to_same_length(y, maxlen)
    if length(y) <= 4
        fill(mean(y), maxlen)
    elseif length(y) == maxlen
        return y
    else
        f = interp1d(1:length(y), y, kind = "cubic")
        f(range(1, length(y), length = maxlen))
    end
end

function load_dataset(dset)
    ts_trn = glob("*/$dset/$(dset)_TRAIN.ts", UEA_UCR)[1]
    ts_tst = replace(ts_trn, "TRAIN" => "TEST")
    h5_trn = replace(ts_trn, ".ts" => ".h5")
    h5_tst = replace(h5_trn, "TRAIN" => "TEST")
    if !isfile(h5_trn) || !isfile(h5_tst)
        X_trn, y_trn = load_from_tsfile_to_dataframe(ts_trn)
        X_tst, y_tst = load_from_tsfile_to_dataframe(ts_tst)
        X = pd.concat([X_trn, X_tst], ignore_index = true)
        y = np.concatenate([y_trn, y_tst])
        seqlens = X["dim_0"].str.len()
        maxlen = seqlens.max()
        if seqlens.nunique() > 1
            for X in [X_trn, X_tst]
                @showprogress for i in X.index, c in X.columns
                    if length(X.at[i, c]) < maxlen
                        X.at[i, c] = to_same_length(X.at[i, c], maxlen)
                    end
                end
            end
        end
        enc = LabelEncoder().fit(y)
        for (X, y, dst) in [(X_trn, y_trn, h5_trn), (X_tst, y_tst, h5_tst)]
            x = permutedims(np.stack(X.apply(np.vstack, axis = 1)), (2, 1, 3))
            y = reshape(enc.transform(y), 1, :)
            x = (x .- mean(x, dims = (2, 3))) ./ std(x, dims = (2, 3))
            h5save(dst, (x = Float32.(x), y = Float32.(y)))
        end
    end
    d_trn = h5open(readmmap, h5_trn)
    d_tst = h5open(readmmap, h5_tst)
    d_trn["x"], d_trn["y"], d_tst["x"], d_tst["y"]
end

UEA_UCR = mkpath(get(ENV, "UEA_UCR_DATA_DIR", expanduser("~/job/UEA_UCR")))
csv = joinpath(UEA_UCR, "result.csv")
base_url = "http://www.timeseriesclassification.com/Downloads/Archives"
for ts in ["Univariate2018_ts.zip", "Multivariate2018_ts.zip"]
    localfile = joinpath(UEA_UCR, ts)
    if !isfile(localfile)
        url = joinpath(base_url, ts)
        run(`wget -c $url -O $localfile`)
    end
    subdir = joinpath(UEA_UCR, ts[1:end-11] * "_ts")
    if !isdir(subdir)
        run(`unzip -d $UEA_UCR $localfile`)
    end
end

grid = [
    "dset" => ["FreezerRegularTrain"] ∪ basename.(glob("*/*", UEA_UCR)),
    "lr" => [0, 1e-5],
    "dropout" => [0.0, 0.2],
    "layer" => ["MLP", "AHLN", "RestNet", "Inception", "TCN", "Rocket", "GRU", "BRU", "nBRU"],
    "hidden_sizes" => ["128", "128,128"],
    "use_batch_norm" => [true, false],
    "use_skip_conn" => [true, false],
    "kernel_size" => [3, 5, 7],
    "recept_field_ratio" => [0.1, 0.3, 0.5, 0.7]
]
@unpack dset, lr, dropout, layer, hidden_sizes, use_batch_norm, use_skip_conn, 
        kernel_size, recept_field_ratio = param = @grid 10000 gridparams(grid)
x_trn, y_trn, x_tst, y_tst = load_dataset(dset)
maximum(y_trn) >= 2 && exit()
recept_field = ceil(Int, size(x_trn, 3) * recept_field_ratio)
out_dim = maximum(y_trn) + 1
model = RnnModel(
    lr = 1e-3, epochs = 100, hidden_sizes = hidden_sizes, loss = "spcce",
    kernel_size = kernel_size, recept_field = recept_field,
    layer = layer, out_dim = out_dim, validation_split = 0
)
RnnLib.fit!(model, x_trn, y_trn)
ŷ_trn = RnnLib.predict(model, x_trn)
ŷ_tst = RnnLib.predict(model, x_tst)
sr = Series(param)
sr["acc_trn"] = accuracy_score(vec(y_trn), vec(ŷ_trn))
sr["acc_tst"] = accuracy_score(vec(y_tst), vec(ŷ_tst))
df = isfile(csv) ? pd.read_csv(csv) : DataFrame()
df = df.append(sr, ignore_index = true).dropna().drop_duplicates()
df.to_csv(csv, index = false)