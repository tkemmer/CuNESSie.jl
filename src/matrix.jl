# TODO
struct SystemMatrix{T} <: AbstractArray{T, 2}
    Ξ       ::CuVector{T}
    elements::CuVector{T}
    numelem ::Int
    params  ::Option{T}
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

@inline function Base.:*(A::SystemMatrix{T}, x::AbstractArray{T, 1}) where T
    Array(matvec(A.elements, A.Ξ, x, A.numelem, A.params))
end

# TODO mul!/5
@inline function LinearAlgebra.mul!(
    Y::AbstractArray{T, 1},
    A::SystemMatrix{T},
    v::AbstractArray{T, 1}
) where T
    Y .= A * v
end

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
