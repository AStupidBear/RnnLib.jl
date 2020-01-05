import numpy as np
from numba import jit

def loss_augmented_inference(r, z, c = 1e-5, ϵ = 0.1):
    T, N = r.shape
    Vᵗ = np.zeros((N, 3), np.float32)
    Ṽᵗ = np.zeros((N, 3), np.float32)
    Q = np.zeros((T, N, 3, 5), np.float32)
    π = np.zeros((T, N), np.int8)
    S = np.zeros(N, np.int8)
    M = np.array([[-1, -1, -1, 0, 1], [-1, 0, 0, 0, 1], [-1, 0, 1, 1, 1]], np.int8)
    # M = np.array([[-1, -1, 0, 0, 1], [-1, 0, 0, 0, 1], [-1, 0, 0, 1, 1]], dtype = 'int8')
    for t in range(T - 1, -1, -1):
        for n in range(N):
            rₙₜ, zₙₜ = r[t, n], z[n, t]
            for s in [-1, 0, 1]:
                for a in range(5):
                    s̃ = np.where(t == T, 0, M[s + 1, a])
                    x, y = np.sign(a - 2), np.abs(a - 2) / 2
                    Q[t, n, s + 1, a] = x * (y * zₙₜ + (1 - y))
                    Q[t, n, s + 1, a] += ϵ * (Ṽᵗ[n, s̃ + 1] + rₙₜ * s̃ - c * abs(s̃ - s))
                    Vᵗ[n, s + 1] = Q[t, n, s + 1, :].max()
        for n in range(N):
            for i in range(3):
                Ṽᵗ[n, i] = Vᵗ[n, i]
    for t in range(T):
        for n in range(N):
            π[t, n] = Q[t, n, S[n] + 1, :].argmax()
            S[n] = M[s + 1, π[t, n]]
    return π

def direct(y_true, y_pred, c=commission, λ=pnl_scale, ϵ = 0.1):
    r, z, c = λ * y_true, y_pred, λ * c
    yw = loss_augmented_inference(r, z, c, 0)
    yϵ = loss_augmented_inference(r, z, c, ϵ)
    return score(z, yw) - score(z, yϵ)

def score(z, y):
    a = K.sign(y - 2)
    b = K.abs(y - 2) / 2
    return a * (b * z + 1 - b)

# %%time
r = np.random.randn(100, 100)
z = 2 * np.random.randn(100, 100)
f = jit(dpopt, nopython=True)
π = f(r, z, 0)
π1 = dpopt(r, z, ϵ = 0)