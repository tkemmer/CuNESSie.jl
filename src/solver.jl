function solve(
         ::Type{NonlocalES},
    model::Model{T, Triangle{T}}
) where T
    elements = CuArray(
        unpack([[e.v1; e.v2; e.v3; e.normal; e.distorig; e.area] for e in model.elements])
    )
    Ξ        = CuArray(unpack([e.center for e in model.elements]))
    numelem  = length(model.elements)

    matvec(
        elements,
        Ξ,
        CuArrays.ones(T, numelem),
        CuArrays.ones(T, numelem),
        CuArrays.ones(T, numelem),
        numelem,
        model.params
    )
end

function matvec(
    elements::CuVector{T},
    Ξ       ::CuVector{T},
    x1      ::CuVector{T},
    x2      ::CuVector{T},
    x3      ::CuVector{T},
    numelem ::Int,
    params  ::Option{T}
) where T
    yuk = yukawa(params)
    _config(kernel) = (threads = 256, blocks = cld(numelem, 256))

    v11 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec11!(v11, elements, Ξ, x1, numelem, yuk)

    v12 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec12!(v12, elements, Ξ, x2, numelem,
        params.εΩ * (1/params.ε∞ - 1/params.εΣ), params.εΩ/params.ε∞, yuk)

    v13 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec13!(v13, elements, Ξ, x3, numelem, params.ε∞/params.εΣ, yuk)

    v21 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec21!(v21, elements, Ξ, x1, numelem)

    v22 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec22!(v22, elements, Ξ, x2, numelem)

    v32 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec32!(v32, elements, Ξ, x2, numelem, params.εΩ/params.ε∞)

    v33 = CuArray{T}(undef, numelem)
    @cuda config=_config matvec33!(v33, elements, Ξ, x3, numelem)

    [v11 .+ v12 .+ v13; v21 .+ v22; v32 .+ v33]
end

function matvec11!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    x1      ::CuDeviceVector{T},
    numelem ::Int,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            (i == j ? T(2π) : T(0)) -
                regularyukawapot_double(Ξ, elements, (i-1) * 3 + 1, (j-1) * 14 + 1, yuk)-
                laplacepot_double(Ξ, elements, (i-1) * 3 + 1, (j-1) * 14 + 1),
            x1[i],
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
    x1      ::CuDeviceVector{T},
    numelem ::Int,
    pref1   ::T,
    pref2   ::T,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            CUDAnative.fma(
                regularyukawapot_single(Ξ, elements, (i-1) * 3 + 1, (j-1) * 14 + 1, yuk),
                pref1,
                pref2 * laplacepot_single(Ξ, elements, (i-1) * 3 + 1, (j-1) * 14 + 1)
            ),
            x1[i],
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
    x1      ::CuDeviceVector{T},
    numelem ::Int,
    pref    ::T,
    yuk     ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            regularyukawapot_double(Ξ, elements, (i-1) * 3 + 1, (j-1) * 14 + 1, yuk),
            x1[i],
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
    x1      ::CuDeviceVector{T},
    numelem ::Int
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            (i == j ? T(2π) : T(0)) +
                laplacepot_double(Ξ, elements, (i-1) * 3 + 1, (j-1) * 14 + 1),
            x1[i],
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
    x1      ::CuDeviceVector{T},
    numelem ::Int
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            laplacepot_single(Ξ, elements, (i-1) * 3 + 1, (j-1) * 14 + 1),
            x1[i],
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
    x1      ::CuDeviceVector{T},
    numelem ::Int,
    pref    ::T
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            laplacepot_single(Ξ, elements, (i-1) * 3 + 1, (j-1) * 14 + 1),
            x1[i],
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
    x1      ::CuDeviceVector{T},
    numelem ::Int
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > numelem
        return nothing
    end

    val = zero(Ξ[1])
    for j in 1:numelem
        val = CUDAnative.fma(
            (i == j ? T(2π) : T(0)) -
                laplacepot_double(Ξ, elements, (i-1) * 3 + 1, (j-1) * 14 + 1),
            x1[i],
            val
        )
    end
    dst[i] = val
    nothing
end
