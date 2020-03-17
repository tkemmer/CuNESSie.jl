function rfenergy(bem::NESSie.BEM.NonlocalBEMResult{T}) where T
    numΞ    = length(bem.model.charges)
    numelem = length(bem.model.elements)
    cuΞ     = Ξ2device([c.pos for c in bem.model.charges])
    cuelms  = elements2device(bem.model.elements)
    cux     = CuArray([bem.u; bem.q])
    dst     = CuArray{T}(undef, numΞ)

    @cuda config=_kcfg(numΞ) _rfenergy_kernel!(dst, cuΞ, cuelms, cux, numΞ, numelem)

    Array(dst) ⋅ [c.val for c in bem.model.charges] * T(110.56123849403735)
end

function _rfenergy_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    x       ::CuDeviceVector{T},
    numΞ    ::Int,
    numelem ::Int
) where T
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > numΞ && return

    ξ = CuPosition(Ξ, i, numΞ)
    val = T(0)
    for j in 1:numelem
        elem = CuTriangle(elements, (j - 1) * 14 + 1)
        val = CUDAnative.fma(-laplacepot_double(ξ, elem), x[j], val)
        val = CUDAnative.fma(laplacepot_single(ξ, elem), x[j + numelem], val)
    end
    dst[i] = val
    nothing
end
