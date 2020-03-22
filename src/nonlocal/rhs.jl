# TODO
struct NonlocalSystemOutputs{T} <: AbstractArray{T, 1}
    β      ::Vector{T}
    numelem::Int
end

function NonlocalSystemOutputs(
    A   ::NonlocalSystemMatrix{T},
    umol::Vector{T},
    qmol::Vector{T}
) where T
    numelem = length(A.elements)
    yuk = yukawa(A.params)
    cfg = _kcfg(numelem)

    cux = CuArray(umol)
    ks  = CuArray{T}(undef, numelem)
    @cuda config=cfg _rhs_k_kernel!(ks, A.Ξ, A.elements, cux,
        one(yuk) - (A.params.εΩ / A.params.εΣ), yuk)

    cux .= CuArray(qmol)
    vs  = CuArray{T}(undef, numelem)
    @cuda config=cfg _rhs_v_kernel!(vs, A.Ξ, A.elements, cux,
        A.params.εΩ * (one(yuk)/A.params.εΣ - one(yuk)/A.params.ε∞),
        A.params.εΩ / A.params.ε∞,
        yuk
    )

    NonlocalSystemOutputs(Array(ks .+ vs), numelem)
end

@inline Base.size(v::NonlocalSystemOutputs{T}) where T = (3 * v.numelem, )

@inline Base.getindex(
    v::NonlocalSystemOutputs{T},
    i::Int
) where T = i ∈ (v.numelem+1):(3*v.numelem) ? zero(T) : Base.getindex(v.β, i)
# FIXME checkounds message

@inline Base.setindex(
    v::NonlocalSystemOutputs{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for ", typeof(v))

function _rhs_k_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    umol    ::CuDeviceVector{T},
    pref    ::T,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    ξ = Ξ[i]
    val = zero(yuk)
    for j in 1:length(elements)
        elem = elements[j]
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
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    qmol    ::CuDeviceVector{T},
    pref1   ::T,
    pref2   ::T,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    ξ = Ξ[i]
    val = zero(yuk)
    for j in 1:length(elements)
        elem = elements[j]
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
