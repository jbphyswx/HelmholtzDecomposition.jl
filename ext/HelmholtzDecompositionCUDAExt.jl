"""
    HelmholtzDecompositionCUDAExt — GPU spectral Helmholtz decomposition via CUFFT.

The Leray projector (`helmholtz_project_spectral!`) is pure fused broadcast and runs
unmodified on `CuArray`s, so the GPU path is: forward `rfft` (CUFFT) → project on the GPU
→ optional inverse `irfft`. This extension wires CUFFT in and resolves an `AutoBackend` to a
`GPUBackend` for `CuArray` inputs.

`helmholtz_decompose_spectral(U::CuArray, grid)` returns the physical GPU velocity split as
a `(; u_rot, u_div, u_harm)` NamedTuple of `CuArray`s. (The full `HelmholtzResult` with
potentials and harmonic diagnostics over the host grid remains a CPU construction; raw
coefficients are available from `helmholtz_project_spectral`.)
"""
module HelmholtzDecompositionCUDAExt

using HelmholtzDecomposition: HelmholtzDecomposition as HD
using CUDA: CUDA
using AbstractFFTs: rfft, irfft

# AutoBackend → GPU for device arrays.
HD._resolve_backend(::HD.AutoBackend, ::CUDA.CuArray) = HD.GPUBackend(:cuda)

# Per-axis angular wavenumbers on the device, axis 1 rfft-reduced.
function _gpu_wavenumbers(::Type{T}, dims::NTuple{N,Int}, spacing::NTuple{N,T}) where {T,N}
    return ntuple(Val(N)) do d
        host = if d == 1
            T[T(2π) * (i - 1) / (dims[1] * spacing[1]) for i in 1:(dims[1] ÷ 2 + 1)]
        else
            Nd = dims[d]
            T[T(2π) * (i - 1 <= Nd ÷ 2 ? i - 1 : i - 1 - Nd) / (Nd * spacing[d]) for i in 1:Nd]
        end
        CUDA.CuArray(host)
    end
end

function HD._spectral_dispatch(
    U::CUDA.CuArray{T},
    grid::HD.StructuredGrid{N,<:HD.CartesianGeometry{N,T}};
    solver = nothing,
    kwargs...,
) where {T,N}
    dims = HD.size_tuple(grid)
    spacing = grid.geometry.spacing

    ĉ1 = rfft(HD._component(U, 1, Val(N)))
    velocity_hat = similar(ĉ1, (size(ĉ1)..., N))
    copyto!(HD._component(velocity_hat, 1, Val(N)), ĉ1)
    for c in 2:N
        copyto!(HD._component(velocity_hat, c, Val(N)), rfft(HD._component(U, c, Val(N))))
    end

    # GPU Leray projection (pure broadcast) → inverse transform to the physical split.
    ks = _gpu_wavenumbers(T, dims, spacing)
    proj = HD.helmholtz_project_spectral(velocity_hat, ks)
    u_rot = similar(U)
    u_div = similar(U)
    for c in 1:N
        copyto!(HD._component(u_rot, c, Val(N)), irfft(HD._component(proj.u_rot, c, Val(N)), dims[1]))
        copyto!(HD._component(u_div, c, Val(N)), irfft(HD._component(proj.u_div, c, Val(N)), dims[1]))
    end
    u_harm = U .- u_rot .- u_div
    return (; u_rot, u_div, u_harm)
end

end # module
