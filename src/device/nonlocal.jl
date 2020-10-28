#=
 TODO try to get rid of these two kernels
=#

function _mul_ld_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    x       ::CuDeviceVector{T}
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    ξ = Ξ[i]
    numelem = length(elements)
    val12 = T(0)
    val3  = T(0)
    for j in 1:numelem
        ld = laplacepot_double(ξ, elements[j])

        val12 = CUDA.fma(ld, x[j], val12)
        val3  = CUDA.fma(ld, x[j + 2numelem], val3)
    end
    dst[i] = -val12
    dst[i + numelem] = val12
    dst[i + 2numelem] = -val3
    nothing
end

function _mul_yd_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    x       ::CuDeviceVector{T},
    pref    ::T,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    ξ = Ξ[i]
    numelem = length(elements)
    val1 = T(0)
    val2 = T(0)
    for j in 1:numelem
        yd = regularyukawapot_double(ξ, elements[j], yuk)

        val1 = CUDA.fma(yd, x[j], val1)
        val2 = CUDA.fma(yd, x[j + 2numelem], val2)
    end
    dst[i] = pref * val2 - val1
    nothing
end
