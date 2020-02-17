function righthandside(
    Ξ       ::CuVector{T},
    elements::CuVector{T},
    umol    ::Vector{T},
    qmol    ::Vector{T},
    numelem ::Int,
    params  ::Option{T}
) where T
    yuk = yukawa(params)
    _config(kernel) = (threads = 256, blocks = cld(numelem, 256))

    cux = CuArray(umol)
    ks  = CuArray{T}(undef, numelem)
    @cuda config=_config _rhs_k_kernel!(ks, elements, Ξ, cux, numelem,
        one(yuk) - (params.εΩ / params.εΣ), yuk)

    cux = CuArray(qmol)
    vs  = CuArray{T}(undef, numelem)
    @cuda config=_config _rhs_v_kernel!(vs, elements, Ξ, cux, numelem,
        params.εΩ * (one(yuk)/params.εΣ - one(yuk)/params.ε∞), params.εΩ / params.ε∞, yuk)

    Array(ks .+ vs)
end

function _rhs_k_kernel!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    umol    ::CuDeviceVector{T},
    numelem ::Int,
    pref    ::T,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
    val = zero(Ξ[1])
    for j in 1:numelem
        elem = CuTriangle(elements, (j - 1) * 14 + 1)
        val = CUDAnative.fma(
            CUDAnative.fma(
                regularyukawapot_double(ξ, elem, yuk),
                pref,
                (i == j ? T(-2π) : T(0)) + laplacepot_double(ξ, elem)
            ),
            umol[j],
            val
        )
    end
    dst[i] = val
    nothing
end

function _rhs_v_kernel!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    qmol    ::CuDeviceVector{T},
    numelem ::Int,
    pref1   ::T,
    pref2   ::T,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
    val = zero(Ξ[1])
    for j in 1:numelem
        elem = CuTriangle(elements, (j - 1) * 14 + 1)
        val = CUDAnative.fma(
            CUDAnative.fma(
                regularyukawapot_single(ξ, elem, yuk),
                pref1,
                -pref2 * laplacepot_single(ξ, elem)
            ),
            qmol[j],
            val
        )
    end
    dst[i] = val
    nothing
end
