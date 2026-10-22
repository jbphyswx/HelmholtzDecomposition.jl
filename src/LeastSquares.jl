# Conditioning of the scattered fits. A fit solves `min ‖A F − vals‖`, `A` the synthesis at the sample
# points; a node set clustered into part of the domain is numerically rank deficient at any mode count,
# with a residual at round-off. `κ(A)` tells the cases apart, and conjugate gradients carries it: its `α`
# and `β` are the Lanczos recurrence for `AᴴA`, whose tridiagonal's extreme eigenvalues approximate those
# of `AᴴA`. A normal-equations solve reaches a relative accuracy of order `κ(A)² eps`, so a fit meets
# `rtol` while `κ² eps ≤ rtol`.

"""
    CGLanczos{V}

The `α` and `β` a conjugate-gradient run produces, per column, kept for [`condition_estimate`](@ref).

`α` and `β` are already computed by the iteration; storing them costs two numbers per step and lets
the conditioning be read off afterwards with no extra transform.
"""
struct CGLanczos{V<:AbstractMatrix}
    α::V        # (maxiter, ncols)
    β::V
    steps::Vector{Int}
end

CGLanczos(::Type{T}, maxiter::Int, ncols::Int) where {T} =
    CGLanczos(zeros(T, maxiter, ncols), zeros(T, maxiter, ncols), zeros(Int, ncols))

@inline function record!(l::CGLanczos, c::Int, k::Int, α::Real, β::Real)
    @inbounds if k <= size(l.α, 1)
        l.α[k, c] = α
        l.β[k, c] = β
        l.steps[c] = k
    end
    return nothing
end

"""
    condition_estimate(l::CGLanczos, c::Int) -> Float64

`κ(A)` for column `c`, from the Lanczos tridiagonal `k` conjugate-gradient steps imply.

`T`'s diagonal is `1/αⱼ + βⱼ₋₁/αⱼ₋₁` and its off-diagonal `√βⱼ/αⱼ`; its extreme eigenvalues
approximate those of `AᴴA`, and `κ(A)` is the square root of their ratio. One step gives `1`, where
there is no ratio to take.
"""
function condition_estimate(l::CGLanczos, c::Int)
    k = l.steps[c]
    k < 2 && return 1.0
    d = Vector{Float64}(undef, k)
    e = Vector{Float64}(undef, k - 1)
    @inbounds for j in 1:k
        αj = l.α[j, c]
        iszero(αj) && return 1.0
        d[j] = j == 1 ? 1 / αj : 1 / αj + l.β[j - 1, c] / l.α[j - 1, c]
        j < k && (e[j] = sqrt(max(l.β[j, c], 0.0)) / αj)
    end
    T = LinearAlgebra.SymTridiagonal(d, e)
    lo = LinearAlgebra.eigmin(T)
    hi = LinearAlgebra.eigmax(T)
    (lo <= 0 || !isfinite(lo) || !isfinite(hi)) && return Inf
    return sqrt(hi / lo)
end

"""
    condition_limit(rtol, T) -> Float64

The `κ(A)` up to which a normal-equations fit delivers `rtol`.

Such a solve is limited to a relative accuracy of order `κ(A)² u`, so `rtol` is reachable while
`κ ≤ √(rtol/u)`. Derived from the tolerance the caller asked for, with no tuned constant in it.
"""
@inline condition_limit(rtol::Real, ::Type{T}) where {T<:AbstractFloat} =
    sqrt(float(rtol) / eps(T))

"""
    warn_conditioning(κ, rtol, limit, npoints, nmodes)

Warn once when `κ` says the node set cannot support `rtol`.

`limit` is the caller's own setting: [`condition_limit`](@ref) by default, `Inf` to say nothing.
"""
function warn_conditioning(κ::Real, rtol::Real, limit::Real, npoints::Integer, nmodes::Integer)
    (isfinite(limit) && κ > limit) || return nothing
    @warn("this point set determines the $nmodes coefficients poorly from its $npoints samples " *
          "(condition number about $(round(κ; sigdigits = 3)), above $(round(limit; sigdigits = 3)) " *
          "where rtol=$rtol stops being reachable). The fit still matches the samples and the " *
          "coefficients it picks are one of many that do. Spread the nodes, reduce the mode " *
          "count, or set `atol` to the noise level so the fit stops once the samples are matched " *
          "to within it.", maxlog = 1)
    return nothing
end
