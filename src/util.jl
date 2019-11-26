
@generated function subslice(x::AbstractArray{T, N}) where {T, N}
    inds = ntuple(i -> (:), N - 1)
    :($inds)
end
subslice(x) = ntuple(i -> (:), ndims(x) - 1)

cview(a, i) = view(a, subslice(a)..., i)

function rebatch(x::AbstractMatrix, batchsize)
    N, T = size(x, 1), size(x, 2)
    n = batchsize ÷ N
    (n <= 1 || T <= n) && return x
    T′, N′ = T ÷ n, N * n
    xv = view(x, :, 1:(T′ * n))
    xp = PermutedDimsArray(xv, [2, 1])
    xr = reshape(xp, T′, N′)
    PermutedDimsArray(xr, [2, 1])
end

function rebatch(x::AbstractArray{<:Any, 3}, batchsize)
    N, T = size(x, 2), size(x, 3)
    n = batchsize ÷ N
    (n <= 1 || T <= n) && return x
    T′, N′ = T ÷ n, N * n
    xv = view(x, :, :, 1:(T′ * n))
    xp = PermutedDimsArray(xv, [1, 3, 2])
    xr = reshape(xp, :, T′, N′)
    PermutedDimsArray(xr, [1, 3, 2])
end

rebatchseq(x::AbstractArray{T, 3}, seq_size) where T = rebatch(x, size(x)[end] ÷ seq_size * size(x)[end - 1])

batchfirst(z::AbstractArray{T, 3}) where T = PermutedDimsArray(z, [1, 3, 2])
batchfirst(z::AbstractVecOrMat) = z