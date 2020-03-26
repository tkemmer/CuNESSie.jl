function _laplace_diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    pot     ::F
) where {T, F <: Function}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    dst[i] = pot(Ξ[i], elements[i])
    nothing
end

for L in [:single, :double]
    @eval begin
        @inline $(Symbol("_laplace_", L, "_diag_kernel!"))(
            dst     ::CuDeviceVector{T},
            Ξ       ::CuPositionVector{T},
            elements::CuTriangleVector{T}
        ) where T = _laplace_diag_kernel!(dst, Ξ, elements, $(Symbol("laplacepot_", L)))
    end
end

function _laplace_mul_kernel!(
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
        val = CUDAnative.fma(pot(ξ, elements[j]), x[j], val)
    end
    dst[i] = val
    nothing
end

for L in [:single, :double]
    @eval begin
        @inline $(Symbol("_laplace_", L, "_mul_kernel!"))(
            dst     ::CuDeviceVector{T},
            Ξ       ::CuPositionVector{T},
            elements::CuTriangleVector{T},
            x       ::CuDeviceVector{T}
        ) where T = _laplace_mul_kernel!(dst, Ξ, elements, x, $(Symbol("laplacepot_", L)))
    end
end

function _reyukawa_diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    yuk     ::T,
    pot     ::F
) where {T, F <: Function}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    dst[i] = pot(Ξ[i], elements[i], yuk)
    nothing
end

for L in [:single, :double]
    @eval begin
        @inline $(Symbol("_reyukawa_", L, "_diag_kernel!"))(
            dst     ::CuDeviceVector{T},
            Ξ       ::CuPositionVector{T},
            elements::CuTriangleVector{T},
            yuk     ::T
        ) where T = _reyukawa_diag_kernel!(dst, Ξ, elements, yuk, $(Symbol("regularyukawapot_", L)))
    end
end

function _reyukawa_mul_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    x       ::CuDeviceVector{T},
    yuk     ::T,
    pot     ::F
) where {T, F <: Function}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    ξ = Ξ[i]
    val = T(0)
    for j in 1:length(elements)
        val = CUDAnative.fma(pot(ξ, elements[j], yuk), x[j], val)
    end
    dst[i] = val
    nothing
end

for L in [:single, :double]
    @eval begin
        @inline $(Symbol("_reyukawa_", L, "_mul_kernel!"))(
            dst     ::CuDeviceVector{T},
            Ξ       ::CuPositionVector{T},
            elements::CuTriangleVector{T},
            x       ::CuDeviceVector{T},
            yuk     ::T
        ) where T = _reyukawa_mul_kernel!(
            dst, Ξ, elements, x, yuk, $(Symbol("regularyukawapot_", L))
        )
    end
end
