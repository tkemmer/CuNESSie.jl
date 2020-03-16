# TODO
struct SystemMatrix{T} <: AbstractArray{T, 2}
    Ξ       ::CuVector{T} # soa
    elements::CuVector{T} # aso
    numelem ::Int
    params  ::Option{T}
end

@inline function SystemMatrix(model::Model{T, Triangle{T}}) where T
    Ξ = CuArray([
        [e.center[1] for e in model.elements];
        [e.center[2] for e in model.elements];
        [e.center[3] for e in model.elements]
    ])
    elements = CuArray(
        unpack([[e.v1; e.v2; e.v3; e.normal; e.distorig; e.area] for e in model.elements])
    )
    SystemMatrix(
        Ξ,
        elements,
        length(model.elements),
        model.params
    )
end

@inline Base.size(A::SystemMatrix{T}) where T = (3 * A.numelem, 3 * A.numelem)

@inline Base.getindex(
    A::SystemMatrix{T},
     ::Int
) where T = error("getindex not defined for ", typeof(A))

@inline Base.setindex!(
    A::SystemMatrix{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for ", typeof(A))

@inline function Base.show(io::IO, ::MIME"text/plain", A::SystemMatrix{T}) where T
    n = 3 * A.numelem
    print(io, "$n×$n SystemMatrix{$T}($(A.numelem) triangles, $(repr(A.params)))")
end

@inline function Base.show(io::IO, A::SystemMatrix{T}) where T
    print(io, "SystemMatrix{$T}($(A.numelem)Δ)")
end

function LinearAlgebra.diag(A::SystemMatrix{T}, k::Int = 0) where T
    k != 0 && error("diag not defined for k != 0")

    _config(kernel) = (threads = 256, blocks = cld(A.numelem, 256))
    dst = CuArray{T}(undef, 3 * A.numelem)
    @cuda config=_config _diag_kernel!(dst, A.Ξ, A.elements, A.numelem, yukawa(A.params))
    Array(dst)
end

function _diag_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    numelem ::Int,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    ξ = CuPosition(Ξ, i, numelem)
    elem = CuTriangle(elements, (i - 1) * 14 + 1)
    ld = laplacepot_double(ξ, elem)
    dst[i]            = T(2π) - regularyukawapot_double(ξ, elem, yuk) - ld
    dst[i + numelem]  = laplacepot_single(ξ, elem)
    dst[i + 2numelem] = T(2π) - ld
    nothing
end

# TODO mul!/5
@inline function LinearAlgebra.mul!(
    Y::AbstractArray{T, 1},
    A::SystemMatrix{T},
    v::AbstractArray{T, 1}
) where T
    Y .= A * v
end

@inline function Base.:*(A::SystemMatrix{T}, x::AbstractArray{T, 1}) where T
    Array(_mul(A.elements, A.Ξ, CuArray(x), A.numelem, A.params))
end

# TODO merge into mul?
function _mul(
    elements::CuVector{T},
    Ξ       ::CuVector{T},
    x       ::CuVector{T},
    numelem ::Int,
    params  ::Option{T}
) where T
    _config(kernel) = (threads = 128, blocks = cld(numelem, 128))
    yuk = yukawa(params)

    ls = CuArray{T}(undef, 3numelem)
    @cuda config=_config _mul_ls_kernel!(ls, Ξ, elements, x, numelem, params.εΩ/params.ε∞)

    ld = CuArray{T}(undef, 3numelem)
    @cuda config=_config _mul_ld_kernel!(ld, Ξ, elements, x, numelem)

    ys = CuArray{T}(undef, numelem)
    @cuda config=_config _mul_ys_kernel!(ys, Ξ, elements, x, numelem,
        params.εΩ * (1/params.ε∞ - 1/params.εΣ), yuk)

    yd = CuArray{T}(undef, numelem)
    @cuda config=_config _mul_yd_kernel!(yd, Ξ, elements, x, numelem,
        params.ε∞ / params.εΣ, yuk)

    [
        T(2π) .* x[1:numelem] .+ ls[1:numelem] .+ ld[1:numelem] .+ ys .+ yd;
        T(2π) .* x[1:numelem] .+ ls[numelem+1:2numelem] .+ ld[numelem+1:2numelem];
        T(2π) .* x[2numelem+1:end] .+ ls[2numelem+1:end] .+ ld[2numelem+1:end]
    ]
end

function _mul_ls_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numelem ::Int,
    pref    ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > numelem && return

    ξ = CuPosition(Ξ, i, numelem)
    val = T(0)
    for j in 1:numelem
        elem = CuTriangle(elements, (j - 1) * 14 + 1)
        val = CUDAnative.fma(laplacepot_single(ξ, elem), x[j + numelem], val)
    end
    dst[i] = pref * val
    dst[i + numelem] = -val
    dst[i + 2numelem] = pref * val
    nothing
end

function _mul_ld_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numelem ::Int
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > numelem && return

    ξ = CuPosition(Ξ, i, numelem)
    val12 = T(0)
    val3  = T(0)
    for j in 1:numelem
        elem = CuTriangle(elements, (j - 1) * 14 + 1)
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
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numelem ::Int,
    pref    ::T,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > numelem && return

    ξ = CuPosition(Ξ, i, numelem)
    val = T(0)
    for j in 1:numelem
        elem = CuTriangle(elements, (j - 1) * 14 + 1)
        val = CUDAnative.fma(regularyukawapot_single(ξ, elem, yuk), x[j + numelem], val)
    end
    dst[i] = pref * val
    nothing
end

function _mul_yd_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numelem ::Int,
    pref    ::T,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > numelem && return

    ξ = CuPosition(Ξ, i, numelem)
    val1 = T(0)
    val2 = T(0)
    for j in 1:numelem
        elem = CuTriangle(elements, (j - 1) * 14 + 1)
        yd = regularyukawapot_double(ξ, elem, yuk)

        val1 = CUDAnative.fma(yd, x[j], val1)
        val2 = CUDAnative.fma(yd, x[j + 2numelem], val2)
    end
    dst[i] = pref * val2 - val1
    nothing
end
