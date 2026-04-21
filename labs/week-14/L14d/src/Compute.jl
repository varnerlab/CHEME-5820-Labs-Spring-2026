"""
    _build_legS_single(h::Int) -> (A, b)

Return the single-channel LegS state matrix `A ∈ ℝ^{h×h}` and input vector
`b ∈ ℝ^{h}` in the 1-based sign convention from Gu et al. 2020 with the stable
form `dx/dt = A x + b u`.
"""
function _build_legS_single(h::Int)
    h >= 1 || error("h must be positive")
    A = zeros(Float64, h, h)
    b = zeros(Float64, h)
    for i in 1:h
        for k in 1:h
            if i > k
                A[i, k] = -sqrt((2i + 1) * (2k + 1))
            elseif i == k
                A[i, k] = -(i + 1)
            end
        end
        b[i] = sqrt(2i + 1)
    end
    return (A, b)
end

"""
    build_legS_matrices_mimo(h::Int, d_in::Int) -> (A, B)

Build the block-diagonal MIMO LegS state matrix `A ∈ ℝ^{(h·d_in)×(h·d_in)}` and
input matrix `B ∈ ℝ^{(h·d_in)×d_in}`. Each input channel drives one independent
`h`-dim LegS filter: `A` is block-diagonal with `d_in` copies of the single-
channel LegS block on its diagonal, and column `j` of `B` is the LegS input
vector placed in block `j` (zero elsewhere).
"""
function build_legS_matrices_mimo(h::Int, d_in::Int)
    h >= 1   || error("h must be positive")
    d_in >= 1 || error("d_in must be positive")
    (A_single, b_single) = _build_legS_single(h)
    H = h * d_in
    A = zeros(Float64, H, H)
    B = zeros(Float64, H, d_in)
    for j in 1:d_in
        rows = ((j - 1) * h + 1):(j * h)
        A[rows, rows] = A_single
        B[rows, j]    = b_single
    end
    return (A, B)
end

"""
    discretize(A, B; Δt, method=:bilinear) -> (Ā, B̄)

Bilinear (Tustin) discretization of the continuous pair `(A, B)` at step `Δt`.

    Ā = (I - Δt/2 A)⁻¹ (I + Δt/2 A)
    B̄ = (I - Δt/2 A)⁻¹ (Δt B)
"""
function discretize(A::AbstractMatrix, B::AbstractMatrix; Δt::Float64, method::Symbol=:bilinear)
    method == :bilinear || error("only :bilinear is supported, got $(method)")
    Δt > 0 || error("Δt must be positive")
    H, W = size(A)
    H == W || error("A must be square, got size $(size(A))")
    size(B, 1) == H || error("B has $(size(B, 1)) rows, expected $(H)")
    I_H = Matrix{Float64}(I, H, H)
    M   = I_H - (Δt / 2) .* A
    Ā   = M \ (I_H + (Δt / 2) .* A)
    B̄   = M \ (Δt .* B)
    return (Ā, B̄)
end

"""
    build(::Type{MyMimoLegSHippoModel}; h, d_in, d_out, Δt, C=nothing, D=nothing, x₀=nothing)
        -> MyMimoLegSHippoModel

Build a MIMO LegS HiPPO model. Constructs the block-diagonal continuous pair
`(A, B)` and its bilinear-discretized counterpart `(Ā, B̄)`. `C` defaults to a
zero matrix of shape `(d_out, h·d_in)` (which `fit_C!` and `fit_C_multi!`
overwrite), `D` defaults to a zero matrix of shape `(d_out, d_in)`, and `x₀`
defaults to zeros.
"""
function build(::Type{MyMimoLegSHippoModel};
    h::Int, d_in::Int, d_out::Int, Δt::Float64,
    C::Union{Nothing, AbstractMatrix} = nothing,
    D::Union{Nothing, AbstractMatrix} = nothing,
    x₀::Union{Nothing, AbstractVector} = nothing,
)
    d_out >= 1 || error("d_out must be positive")
    Δt > 0 || error("Δt must be positive")
    (A, B) = build_legS_matrices_mimo(h, d_in)
    (Ā, B̄) = discretize(A, B; Δt = Δt)
    H = h * d_in
    Cmat = C === nothing ? zeros(Float64, d_out, H) : Matrix{Float64}(C)
    Dmat = D === nothing ? zeros(Float64, d_out, d_in) : Matrix{Float64}(D)
    x0   = x₀ === nothing ? zeros(Float64, H) : collect(Float64, x₀)
    size(Cmat) == (d_out, H)    || error("C has shape $(size(Cmat)), expected ($(d_out), $(H))")
    size(Dmat) == (d_out, d_in) || error("D has shape $(size(Dmat)), expected ($(d_out), $(d_in))")
    length(x0) == H             || error("x₀ has length $(length(x0)), expected $(H)")
    return MyMimoLegSHippoModel(h, d_in, d_out, Δt, A, B, Ā, B̄, Cmat, Dmat, x0)
end

"""
    rollout(model::MyMimoLegSHippoModel, U::AbstractMatrix) -> X

Run the discrete-time recursion `xₜ = Ā xₜ₋₁ + B̄ uₜ` with `x₀ = model.x₀` on
the input matrix `U ∈ ℝ^{T×d_in}` and return the hidden-state matrix
`X ∈ ℝ^{T×(h·d_in)}` whose `t`-th row is `xₜᵀ`.
"""
function rollout(model::MyMimoLegSHippoModel, U::AbstractMatrix)
    T, d_in = size(U)
    d_in == model.d_in || error("U has $(d_in) columns, expected $(model.d_in)")
    H = model.h * model.d_in
    X = zeros(Float64, T, H)
    x = copy(model.x₀)
    @inbounds for t in 1:T
        u = @view U[t, :]
        x = model.Ā * x + model.B̄ * u
        X[t, :] = x
    end
    return X
end

"""
    solve(model::MyMimoLegSHippoModel, U::AbstractMatrix) -> (X, Y)

Roll the model forward on the input matrix `U` and apply the current readout
to produce `Y ∈ ℝ^{T×d_out}` with `Yₜ = C xₜ + D uₜ`.
"""
function solve(model::MyMimoLegSHippoModel, U::AbstractMatrix)
    X = rollout(model, U)
    Y = X * model.C' .+ U * model.D'
    return (X, Y)
end

"""
    fit_C!(model::MyMimoLegSHippoModel, U::AbstractMatrix, Y::AbstractMatrix; λ=1e-4) -> model

Fit the readout `C` by closed-form ridge regression on a single rollout of
`U`, targeting `Y`. Solves

    Cᵀ = (XᵀX + λ I)⁻¹ Xᵀ (Y - U Dᵀ).

Mutates `model.C` and returns `model`.
"""
function fit_C!(model::MyMimoLegSHippoModel, U::AbstractMatrix, Y::AbstractMatrix; λ::Float64 = 1.0e-4)
    λ >= 0 || error("λ must be nonnegative")
    T_u, d_in = size(U)
    T_y, d_out = size(Y)
    T_u == T_y         || error("U has $(T_u) rows, Y has $(T_y) rows; must match")
    d_in == model.d_in || error("U has $(d_in) columns, expected $(model.d_in)")
    d_out == model.d_out || error("Y has $(d_out) columns, expected $(model.d_out)")
    X = rollout(model, U)
    Y_res = Y .- U * model.D'
    H = model.h * model.d_in
    G = X' * X + λ .* Matrix{Float64}(I, H, H)
    rhs = X' * collect(Float64, Y_res)
    CT = G \ rhs
    model.C = Matrix{Float64}(CT')
    return model
end

"""
    fit_C_multi!(model::MyMimoLegSHippoModel,
                  U_list::AbstractVector{<:AbstractMatrix},
                  Y_list::AbstractVector{<:AbstractMatrix};
                  λ::Float64 = 1.0e-4) -> model

Multi-trajectory ridge fit of the readout `C`. For each `(U_i, Y_i)` pair, the
SSM is rolled out starting from `x₀` (default zeros, so every trajectory gets
its own fresh hidden state), the hidden-state matrices are concatenated row-
wise into a single design matrix, and the closed-form normal equations

    Cᵀ = (XᵀX + λ I)⁻¹ Xᵀ (Y - U Dᵀ)

are solved once over all trajectories. Use this when the training data is a
list of independent sequences (for example, one bioreactor trajectory per
feed policy) rather than one long continuous sequence.
"""
function fit_C_multi!(model::MyMimoLegSHippoModel,
    U_list::AbstractVector{<:AbstractMatrix},
    Y_list::AbstractVector{<:AbstractMatrix};
    λ::Float64 = 1.0e-4,
)
    λ >= 0 || error("λ must be nonnegative")
    length(U_list) == length(Y_list) || error("U_list and Y_list must have the same length")
    !isempty(U_list) || error("U_list must be nonempty")

    H = model.h * model.d_in
    # Accumulate XᵀX and Xᵀ(Y - U Dᵀ) across trajectories without materializing
    # a full concatenated X, so memory stays bounded even for many trajectories.
    G   = λ .* Matrix{Float64}(I, H, H)
    rhs = zeros(Float64, H, model.d_out)

    for (i, (U, Y)) in enumerate(zip(U_list, Y_list))
        T_u, d_in = size(U)
        T_y, d_out = size(Y)
        T_u == T_y         || error("trajectory $(i): U has $(T_u) rows, Y has $(T_y)")
        d_in == model.d_in || error("trajectory $(i): U has $(d_in) cols, expected $(model.d_in)")
        d_out == model.d_out || error("trajectory $(i): Y has $(d_out) cols, expected $(model.d_out)")

        X     = rollout(model, U)
        Y_res = Y .- U * model.D'
        G   .+= X' * X
        rhs .+= X' * collect(Float64, Y_res)
    end

    CT = G \ rhs
    model.C = Matrix{Float64}(CT')
    return model
end

"""
    autoregressive_rollout(model::MyMimoLegSHippoModel, x_state0::AbstractVector,
                            cond::AbstractVector, T::Int) -> Matrix{Float64}

Generate a length-`T` state trajectory autoregressively from the trained SSM.
At every step the model input is the concatenation `[x_state; cond]`, where
`x_state` is the initial state at step 1 and the model's own prediction from
step 2 onward. The per-step output `yₜ = C xₜ + D uₜ` is interpreted as the
predicted next state and used as `x_state` for the next step.

Returns a `(T × length(x_state0))` matrix whose rows are the predicted states
at time steps `t = 1, …, T`. Row 1 is always the supplied initial state so
downstream plotting lines up with the ground-truth trajectory.
"""
function autoregressive_rollout(model::MyMimoLegSHippoModel,
    x_state0::AbstractVector{<:Real},
    cond::AbstractVector{<:Real},
    T::Int,
)
    T >= 1 || error("T must be positive")
    n_state = length(x_state0)
    n_cond  = length(cond)
    n_state + n_cond == model.d_in ||
        error("state ($(n_state)) + cond ($(n_cond)) must equal model.d_in = $(model.d_in)")
    n_state == model.d_out ||
        error("length(x_state0) = $(n_state) must equal model.d_out = $(model.d_out)")

    preds = zeros(Float64, T, n_state)
    preds[1, :] = collect(Float64, x_state0)

    h_state = copy(model.x₀)
    u       = zeros(Float64, model.d_in)
    cond_f  = collect(Float64, cond)

    @inbounds for t in 1:(T - 1)
        # Input at step t: current predicted state + feed-policy conditioning
        for j in 1:n_state
            u[j] = preds[t, j]
        end
        for j in 1:n_cond
            u[n_state + j] = cond_f[j]
        end
        h_state = model.Ā * h_state + model.B̄ * u
        y_t     = model.C * h_state .+ model.D * u
        # Interpret yₜ as the predicted state at time t+1
        for j in 1:n_state
            preds[t + 1, j] = y_t[j]
        end
    end
    return preds
end

"""
    predict(model::MyMimoLegSHippoModel, U::AbstractMatrix; warmup::Int=0) -> (X, Y)

Roll the model on a new input matrix `U` using the currently trained `C` and
`D`. If `warmup > 0`, the first `warmup` rows of `X` and `Y` are discarded.
"""
function predict(model::MyMimoLegSHippoModel, U::AbstractMatrix; warmup::Int = 0)
    warmup >= 0 || error("warmup must be nonnegative")
    (X, Y) = solve(model, U)
    if warmup > 0
        warmup < size(U, 1) || error("warmup = $(warmup) must be smaller than length $(size(U, 1))")
        return (X[(warmup + 1):end, :], Y[(warmup + 1):end, :])
    end
    return (X, Y)
end
