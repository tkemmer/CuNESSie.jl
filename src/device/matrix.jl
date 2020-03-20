function _mul_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    dims    ::NTuple{2, Int},
    pot     ::F
) where {T, F <: Function}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > dims[1] && return

    ξ = CuPosition(Ξ, i)
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
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    dims    ::NTuple{2, Int}
) where T = _mul_kernel!(dst, Ξ, elements, x, dims, laplacepot_single)

@inline _laplace_double_mul_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    dims    ::NTuple{2, Int}
) where T = _mul_kernel!(dst, Ξ, elements, x, dims, laplacepot_double)
