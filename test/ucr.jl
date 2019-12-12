using RnnLib
using MLSuiteBase
using PyCall
using PyCallUtils
using HDF5
using HDF5Utils
using ProgressMeter
using Random
using Statistics
using Printf
using Test

ENV["USE_GPU"] = 0

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
    ts_trn = joinpath(ENV["UCR_DATA_DIR"], dset, dset * "_TRAIN")
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
            h5save(dst, (x = x, y = y))
        end
    end
    d_trn = h5open(readmmap, h5_trn)
    d_tst = h5open(readmmap, h5_tst)
    d_trn[:x], d_trn[:y], d_tst[:x], d_tst[:y] 
end

UEA_UCR = mkpath(get(ENV, "UCR_DATA_DIR", expanduser("~/.data/UEA_UCR")))
base_url = "http://www.timeseriesclassification.com/Downloads/Archives"
for ts in ["Univariate2018_ts.zip", "Multivariate2018_ts.zip"]
    file = joinpath(UEA_UCR, "..", ts)
    if !isfile(file)
        url = joinpath(base_url, ts)
        download(url, file)
        run(`unzip -d $UEA_UCR $file`)
        subdir = ts[1:end-11] * "_ts"
        for dir in readir(joinpath(UEA_UCR, subdir))
            mv(dir, joinpath(UEA_UCR, dir))
        end
    end
end

model = RnnModel(loss = "spcce", epochs = 200)
for dset in readdir(UEA_UCR)
    for layer in ["MLP", "ResNet", "Inception", "Rocket"]
        x_trn, y_trn, x_tst, y_tst = load_dataset(dset)
        @pack! model = layer
        RnnLib.fit!(model, x_trn, y_trn)
        ŷ_trn = RnnLib.predict(x_tst)
        ŷ_tst = RnnLib.predict(x_tst)
        acc_trn = accuracy_score(vec(y_trn), vec(ŷ_trn))
        acc_tst = accuracy_score(vec(y_tst), vec(ŷ_tst))
        @printf("%s\tacc_trn:%.2f\tacc_tst", dset, acc_trn, acc_tst)
    end
end