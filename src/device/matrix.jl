function _diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuDeviceVector{T},
    dims    ::Int,
    pot     ::F
) where {T, F <: Function}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > dims && return

    ξ      = Ξ[i]
    elem   = CuTriangle(elements, i)
    dst[i] = pot(ξ, elem)
    nothing
end

@inline _laplace_single_diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuDeviceVector{T},
    dims    ::Int
) where T = _diag_kernel!(dst, Ξ, elements, dims, laplacepot_single)

@inline _laplace_double_diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuDeviceVector{T},
    dims    ::Int
) where T = _diag_kernel!(dst, Ξ, elements, dims, laplacepot_double)

function _mul_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    dims    ::NTuple{2, Int},
    pot     ::F
) where {T, F <: Function}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > dims[1] && return

    ξ = Ξ[i]
    val = T(0)
    for j in 1:dims[2]
        elem = CuTriangle(elements, j)
        val = CUDAnative.fma(pot(ξ, elem), x[j], val)
    end
    dst[i] = val
    nothing
end

@inline _laplace_single_mul_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    dims    ::NTuple{2, Int}
) where T = _mul_kernel!(dst, Ξ, elements, x, dims, laplacepot_single)

@inline _laplace_double_mul_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    dims    ::NTuple{2, Int}
) where T = _mul_kernel!(dst, Ξ, elements, x, dims, laplacepot_double)
