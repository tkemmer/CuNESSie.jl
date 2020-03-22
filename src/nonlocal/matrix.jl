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

    dst = CuArray{T}(undef, size(A, 1))
    @cuda config=_kcfg(size(A, 1)) _diag_kernel!(dst, A.Ξ, A.elements, yukawa(A.params))
    Array(dst)
end

function _diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    ξ = Ξ[i]
    elem = elements[i]
    numelem = length(elements)
    ld = laplacepot_double(ξ, elem)
    dst[i]            = T(2π) - regularyukawapot_double(ξ, elem, yuk) - ld
    dst[i + numelem]  = laplacepot_single(ξ, elem)
    dst[i + 2numelem] = T(2π) - ld
    nothing
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
    yuk = yukawa(params)

    ls = CuArray{T}(undef, 3numelem)
    @cuda config=cfg _mul_ls_kernel!(ls, Ξ, elements, x, params.εΩ/params.ε∞)

    ld = CuArray{T}(undef, 3numelem)
    @cuda config=cfg _mul_ld_kernel!(ld, Ξ, elements, x)

    ys = CuArray{T}(undef, numelem)
    @cuda config=cfg _mul_ys_kernel!(ys, Ξ, elements, x,
        params.εΩ * (1/params.ε∞ - 1/params.εΣ), yuk)

    yd = CuArray{T}(undef, numelem)
    @cuda config=cfg _mul_yd_kernel!(yd, Ξ, elements, x, params.ε∞/params.εΣ, yuk)

    [
        T(2π) .* x[1:numelem] .+ ls[1:numelem] .+ ld[1:numelem] .+ ys .+ yd;
        T(2π) .* x[1:numelem] .+ ls[numelem+1:2numelem] .+ ld[numelem+1:2numelem];
        T(2π) .* x[2numelem+1:end] .+ ls[2numelem+1:end] .+ ld[2numelem+1:end]
    ]
end

function _mul_ls_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuPositionVector{T},
    elements::CuTriangleVector{T},
    x       ::CuDeviceVector{T},
    pref    ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > length(Ξ) && return

    ξ = Ξ[i]
    numelem = length(elements)
    val = T(0)
    for j in 1:numelem
        elem = elements[j]
        val = CUDAnative.fma(laplacepot_single(ξ, elem), x[j + numelem], val)
    end
    dst[i] = pref * val
    dst[i + numelem] = -val
    dst[i + 2numelem] = pref * val
    nothing
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

function _mul_ys_kernel!(
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
    val = T(0)
    for j in 1:numelem
        elem = elements[j]
        val = CUDAnative.fma(regularyukawapot_single(ξ, elem, yuk), x[j + numelem], val)
    end
    dst[i] = pref * val
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
