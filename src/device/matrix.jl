function _diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    pot     ::F
) where {T, F <: Function}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    ξ      = Ξ[i]
    elem   = elements[i]
    dst[i] = pot(ξ, elem)
    nothing
end

@inline _laplace_single_diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T}
) where T = _diag_kernel!(dst, Ξ, elements, laplacepot_single)

@inline _laplace_double_diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T}
) where T = _diag_kernel!(dst, Ξ, elements, laplacepot_double)

function _mul_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    x       ::CuDeviceVector{T},
    pot     ::F
) where {T, F <: Function}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    ξ = Ξ[i]
    val = T(0)
    for j in 1:length(elements)
        elem = elements[j]
        val = CUDAnative.fma(pot(ξ, elem), x[j], val)
    end
    dst[i] = val
    nothing
end

@inline _laplace_single_mul_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    x       ::CuDeviceVector{T}
) where T = _mul_kernel!(dst, Ξ, elements, x, laplacepot_single)

@inline _laplace_double_mul_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    x       ::CuDeviceVector{T}
) where T = _mul_kernel!(dst, Ξ, elements, x, laplacepot_double)
