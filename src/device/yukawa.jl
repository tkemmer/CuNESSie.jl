for T in [:Float64, :Float32]
    varname = Symbol("_radonpts_", T)
    @eval begin
        const $(varname) = (
            num = 7,
            x = $(T).((1/3, (6+√15)/21, (9-2*√15)/21, (6+√15)/21,
                (6-√15)/21, (9+2*√15)/21, (6-√15)/21)),
            y = $(T).((1/3, (9-2*√15)/21, (6+√15)/21, (6+√15)/21,
                (9+2*√15)/21, (6-√15)/21, (6-√15)/21)),
            w = $(T).((9/80, (155+√15)/2400, (155+√15)/2400, (155+√15)/2400,
                (155-√15)/2400, (155-√15)/2400, (155-√15)/2400))
        )
        _radonpoints(::$(T)) = $(varname)
    end
end

@inline function _project2d(
    elem::CuTriangle{T},
    rx  ::T,
    ry  ::T
) where T
    (
        x = CUDAnative.fma(
            elem.v2.x - elem.v1.x,
            rx,
            CUDAnative.fma(elem.v3.x - elem.v1.x, ry, elem.v1.x)
        ),
        y = CUDAnative.fma(
            elem.v2.y - elem.v1.y,
            rx,
            CUDAnative.fma(elem.v3.y - elem.v1.y , ry, elem.v1.y)
        ),
        z = CUDAnative.fma(
            elem.v2.z - elem.v1.z,
            rx,
            CUDAnative.fma(elem.v3.z - elem.v1.z, ry, elem.v1.z)
        )
    )
end

@inline function _refside(
    ξ   ::CuPosition{T},
    elem::CuTriangle{T},
    rx  ::T,
    ry  ::T
) where T
    _sub(_project2d(elem, rx, ry), ξ)
end

@inline function _regularyukawapot_single(rn ::T, yuk::T) where T
    rn < _etol(rn) && return -yuk
    sn = yuk * rn
    rn ≥ 0.1 ?
        CUDAnative.expm1(-sn) / rn :
        _regularyukawapot_single_series(sn) / rn
end

function _regularyukawapot_single_series(sn::T) where T
    term = -sn
    tol  = _etol(sn) * _abs(term)
    tsum = zero(sn)
    for i in 1:15
        _abs(term) <= tol && break
        tsum += term
        term *= -sn / (i+1)
    end
    tsum
end

@inline function _regularyukawapot_double(rn ::T, cs ::T, yuk::T) where T
    rn < _etol(rn) && return yuk * yuk / T(3.4641016151377544)
    sn = yuk * rn
    sn ≥ 0.1 ?
        (one(rn) - (one(rn) + sn) * CUDAnative.exp(-sn)) * cs / rn / rn / rn :
        _regularyukawapot_double_series(sn) * cs / rn / rn / rn
end

function _regularyukawapot_double_series(sn::T) where T
    term = 2 \ sn * sn
    tol  = _etol(sn) * _abs(term)
    tsum = zero(sn)
    for i in 2:16
        _abs(term) <= tol && break
        tsum = CUDAnative.fma(term, T(i-1), tsum)
        term *= -sn / (i+1)
    end
    tsum
end

function regularyukawapot_single(
    ξ   ::CuPosition{T},
    elem::CuTriangle{T},
    yuk ::T
) where T
    pts = _radonpoints(yuk)
    pot = _regularyukawapot_single

    val = zero(yuk)
    val = CUDAnative.fma(pot(_norm(_refside(ξ, elem, pts.x[1], pts.y[1])), yuk), pts.w[1], val)
    val = CUDAnative.fma(pot(_norm(_refside(ξ, elem, pts.x[2], pts.y[2])), yuk), pts.w[2], val)
    val = CUDAnative.fma(pot(_norm(_refside(ξ, elem, pts.x[3], pts.y[3])), yuk), pts.w[3], val)
    val = CUDAnative.fma(pot(_norm(_refside(ξ, elem, pts.x[4], pts.y[4])), yuk), pts.w[4], val)
    val = CUDAnative.fma(pot(_norm(_refside(ξ, elem, pts.x[5], pts.y[5])), yuk), pts.w[5], val)
    val = CUDAnative.fma(pot(_norm(_refside(ξ, elem, pts.x[6], pts.y[6])), yuk), pts.w[6], val)
    val = CUDAnative.fma(pot(_norm(_refside(ξ, elem, pts.x[7], pts.y[7])), yuk), pts.w[7], val)
    val * 2 * elem.area
end

function regularyukawapot_double(
    ξ   ::CuPosition{T},
    elem::CuTriangle{T},
    yuk ::T
) where T
    pts = _radonpoints(yuk)
    pot = _regularyukawapot_double
    nd  = ref -> (_norm(ref), _dot(ref, elem.normal))

    val = zero(yuk)
    val = CUDAnative.fma(pot(nd(_refside(ξ, elem, pts.x[1], pts.y[1]))..., yuk), pts.w[1], val)
    val = CUDAnative.fma(pot(nd(_refside(ξ, elem, pts.x[2], pts.y[2]))..., yuk), pts.w[2], val)
    val = CUDAnative.fma(pot(nd(_refside(ξ, elem, pts.x[3], pts.y[3]))..., yuk), pts.w[3], val)
    val = CUDAnative.fma(pot(nd(_refside(ξ, elem, pts.x[4], pts.y[4]))..., yuk), pts.w[4], val)
    val = CUDAnative.fma(pot(nd(_refside(ξ, elem, pts.x[5], pts.y[5]))..., yuk), pts.w[5], val)
    val = CUDAnative.fma(pot(nd(_refside(ξ, elem, pts.x[6], pts.y[6]))..., yuk), pts.w[6], val)
    val = CUDAnative.fma(pot(nd(_refside(ξ, elem, pts.x[7], pts.y[7]))..., yuk), pts.w[7], val)
    val * 2 * elem.area
end
