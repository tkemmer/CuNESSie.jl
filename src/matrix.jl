# TODO
struct SystemMatrix{T} <: AbstractArray{T, 2}
    Ξ       ::CuVector{T}
    elements::CuVector{T}
    numelem ::Int
    params  ::Option{T}
end

@inline function SystemMatrix(model::Model{T, Triangle{T}}) where T
    elements = CuArray(
        unpack([[e.v1; e.v2; e.v3; e.normal; e.distorig; e.area] for e in model.elements])
    )
    SystemMatrix(
        CuArray(unpack([e.center for e in model.elements])),
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

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
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
    _config(kernel) = (threads = 256, blocks = cld(numelem, 256))
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

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
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

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
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

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
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

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
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

# deprecated
function matvec(
    elements::CuVector{T},
    Ξ       ::CuVector{T},
    x       ::AbstractArray{T, 1},
    numelem ::Int,
    params  ::Option{T}
) where T
    yuk = yukawa(params)
    _config(kernel) = (threads = 256, blocks = cld(numelem, 256))

    cux = CuArray(x[1:numelem])
    v11 = CuArray{T}(undef, numelem)
    v21 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec11!(v11, elements, Ξ, cux, numelem, yuk)
    @cuda config=_config matvec21!(v21, elements, Ξ, cux, numelem)

    cux = CuArray(x[(numelem+1):(2numelem)])
    v12 = CuArray{T}(undef, numelem)
    v22 = CuArray{T}(undef, numelem)
    v32 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec12!(v12, elements, Ξ, cux, numelem,
        params.εΩ * (1/params.ε∞ - 1/params.εΣ), params.εΩ/params.ε∞, yuk)
    @cuda config=_config matvec22!(v22, elements, Ξ, cux, numelem)
    @cuda config=_config matvec32!(v32, elements, Ξ, cux, numelem, params.εΩ/params.ε∞)

    cux = CuArray(x[(2numelem+1):(3numelem)])
    v13 = CuArray{T}(undef, numelem)
    v33 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec13!(v13, elements, Ξ, cux, numelem, params.ε∞/params.εΣ, yuk)
    @cuda config=_config matvec33!(v33, elements, Ξ, cux, numelem)

    [v11 .+ v12 .+ v13; v21 .+ v22; v32 .+ v33]
end

# deprecated
function matvec11!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numelem ::Int,
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
            (i == j ? T(2π) : T(0)) -
                regularyukawapot_double(ξ, elem, yuk) -
                laplacepot_double(ξ, elem),
            x[j],
            val
        )
    end
    dst[i] = val
    nothing
end

# deprecated
function matvec12!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
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
                pref2 * laplacepot_single(ξ, elem)
            ),
            x[j],
            val
        )
    end
    dst[i] = val
    nothing
end

# deprecated
function matvec13!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
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
        val = CUDAnative.fma(
            regularyukawapot_double(ξ, CuTriangle(elements, (j - 1) * 14 + 1), yuk),
            x[j],
            val
        )
    end
    dst[i] = pref * val
    nothing
end

# deprecated
function matvec21!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numelem ::Int
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            (i == j ? T(2π) : T(0)) +
                laplacepot_double(ξ, CuTriangle(elements, (j - 1) * 14 + 1)),
            x[j],
            val
        )
    end
    dst[i] = val
    nothing
end

# deprecated
function matvec22!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numelem ::Int
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            laplacepot_single(ξ, CuTriangle(elements, (j - 1) * 14 + 1)),
            x[j],
            val
        )
    end
    dst[i] = -val
    nothing
end

# deprecated
function matvec32!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numelem ::Int,
    pref    ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            laplacepot_single(ξ, CuTriangle(elements, (j - 1) * 14 + 1)),
            x[j],
            val
        )
    end
    dst[i] = pref * val
    nothing
end

# deprecated
function matvec33!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numelem ::Int
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    ξ = CuPosition(Ξ, (i - 1) * 3 + 1)
    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            (i == j ? T(2π) : T(0)) -
                laplacepot_double(ξ, CuTriangle(elements, (j - 1) * 14 + 1)),
            x[j],
            val
        )
    end
    dst[i] = val
    nothing
end
