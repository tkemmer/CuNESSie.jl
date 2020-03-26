# TODO
struct NonlocalSystemMatrix{T} <: AbstractArray{T, 2}
    Ξ       ::PositionVector{T} # soa
    elements::TriangleVector{T} # aso
    params  ::Option{T}
end

@inline function NonlocalSystemMatrix(model::Model{T, Triangle{T}}) where T
    NonlocalSystemMatrix(
        PositionVector([e.center for e in model.elements]),
        TriangleVector(model.elements),
        model.params
    )
end

@inline Base.size(A::NonlocalSystemMatrix{T}) where T = (3 * length(A.Ξ), 3 * length(A.elements))

@inline Base.getindex(
    A::NonlocalSystemMatrix{T},
     ::Int
) where T = error("getindex not defined for ", typeof(A))

@inline Base.setindex!(
    A::NonlocalSystemMatrix{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for ", typeof(A))

@inline function Base.show(io::IO, ::MIME"text/plain", A::NonlocalSystemMatrix{T}) where T
    print(io, join(size(A), "×"), " ", typeof(A), "($(repr(A.params)))")
end

@inline function Base.show(io::IO, A::NonlocalSystemMatrix{T}) where T
    print(io, typeof(A))
end

function LinearAlgebra.diag(A::NonlocalSystemMatrix{T}, k::Int = 0) where T
    k != 0 && error("diag not defined for k != 0")
    Kʸ = ReYukawaPotentialMatrix{DoubleLayer}(A.Ξ, A.elements, yukawa(A.params))
    V  = LaplacePotentialMatrix{SingleLayer}(A.Ξ, A.elements)
    σ  = T(2π) .* ones(T, size(A.Ξ, 1))
    [ σ .- diag(Kʸ); diag(V) ; σ ]
end

# TODO mul!/5
@inline function LinearAlgebra.mul!(
    Y::AbstractArray{T, 1},
    A::NonlocalSystemMatrix{T},
    v::AbstractArray{T, 1}
) where T
    Y .= A * v
end

@inline function Base.:*(A::NonlocalSystemMatrix{T}, x::AbstractArray{T, 1}) where T
    Array(_mul(A.Ξ, A.elements, CuArray(x), A.params))
end

# TODO merge into mul?
function _mul(
    Ξ       ::PositionVector{T},
    elements::TriangleVector{T},
    x       ::CuVector{T},
    params  ::Option{T}
) where T
    numelem = length(elements)
    cfg = _kcfg(numelem)
    εΩ  = params.εΩ
    εΣ  = params.εΣ
    ε∞  = params.ε∞
    yuk = yukawa(params)

    V  = LaplacePotentialMatrix{SingleLayer}(Ξ, elements)
    Vʸ = ReYukawaPotentialMatrix{SingleLayer}(Ξ, elements, yuk)

    u  = x[1:numelem]
    q  = x[numelem+1:2numelem]
    w  = x[2numelem+1:3numelem]

    ls = V * q
    ys = (εΩ * (1/ε∞ - 1/εΣ)) .* (Vʸ * q)

    ld = CuArray{T}(undef, 3numelem)
    @cuda config=cfg _mul_ld_kernel!(ld, Ξ, elements, x)

    yd = CuArray{T}(undef, numelem)
    @cuda config=cfg _mul_yd_kernel!(yd, Ξ, elements, x, ε∞/εΣ, yuk)

    [
        T(2π) .* u .+ (εΩ/ε∞) .* ls .+ ld[1:numelem] .+ ys .+ yd;
        T(2π) .* u .- ls .+ ld[numelem+1:2numelem];
        T(2π) .* w .+ (εΩ/ε∞) .* ls .+ ld[2numelem+1:end]
    ]
end

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
        elem = elements[j]
        ld = laplacepot_double(ξ, elem)

        val12 = CUDAnative.fma(ld, x[j], val12)
        val3  = CUDAnative.fma(ld, x[j + 2numelem], val3)
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
        elem = elements[j]
        yd = regularyukawapot_double(ξ, elem, yuk)

        val1 = CUDAnative.fma(yd, x[j], val1)
        val2 = CUDAnative.fma(yd, x[j + 2numelem], val2)
    end
    dst[i] = pref * val2 - val1
    nothing
end
