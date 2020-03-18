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
    sn = yuk * rn
    rn < _etol(rn) ?
        -yuk : rn >= 0.1 ?
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
    sn = yuk * rn

    rn < _etol(rn) ?
        yuk * yuk / T(3.4641016151377544) : sn >= 0.1 ?
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
    unit = one(yuk)
    s15  = _sqrt(unit * 15)
    pot  = _regularyukawapot_single

    val = zero(yuk)
    val = CUDAnative.fma(
        pot(_norm(_refside(ξ, elem, unit/3, unit/3)), yuk), unit * 9 / 80, val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(ξ, elem, (6+s15)/21, (9-2*s15)/21)), yuk), (155 + s15)/2400, val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(ξ, elem, (9-2*s15)/21, (6+s15)/21)), yuk), (155 + s15)/2400,val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(ξ, elem, (6+s15)/21, (6+s15)/21)), yuk), (155 + s15)/2400, val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(ξ, elem, (6-s15)/21, (9+2*s15)/21)), yuk), (155 - s15)/2400, val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(ξ, elem, (9+2*s15)/21, (6-s15)/21)), yuk), (155 - s15)/2400, val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(ξ, elem, (6-s15)/21, (6-s15)/21)), yuk), (155 - s15)/2400, val
    )
    val * 2 * elem.area
end

function regularyukawapot_double(
    ξ   ::CuPosition{T},
    elem::CuTriangle{T},
    yuk ::T
) where T
    unit = one(yuk)
    s15  = _sqrt(unit * 15)
    pot  = _regularyukawapot_double

    val = zero(yuk)
    ref = _refside(ξ, elem, unit/3, unit/3)
    val = CUDAnative.fma(
        pot(_norm(ref), _dot(ref, elem.normal), yuk), unit * 9 / 80, val
    )
    ref = _refside(ξ, elem, (6+s15)/21, (9-2*s15)/21)
    val = CUDAnative.fma(
        pot(_norm(ref), _dot(ref, elem.normal), yuk), (155 + s15)/2400, val
    )
    ref = _refside(ξ, elem, (9-2*s15)/21, (6+s15)/21)
    val = CUDAnative.fma(
        pot(_norm(ref), _dot(ref, elem.normal), yuk), (155 + s15)/2400, val
    )
    ref = _refside(ξ, elem, (6+s15)/21, (6+s15)/21)
    val = CUDAnative.fma(
        pot(_norm(ref), _dot(ref, elem.normal), yuk), (155 + s15)/2400, val
    )
    ref = _refside(ξ, elem, (6-s15)/21, (9+2*s15)/21)
    val = CUDAnative.fma(
        pot(_norm(ref), _dot(ref, elem.normal), yuk), (155 - s15)/2400, val
    )
    ref = _refside(ξ, elem, (9+2*s15)/21, (6-s15)/21)
    val = CUDAnative.fma(
        pot(_norm(ref), _dot(ref, elem.normal), yuk), (155 - s15)/2400, val
    )
    ref = _refside(ξ, elem, (6-s15)/21, (6-s15)/21)
    val = CUDAnative.fma(
        pot(_norm(ref), _dot(ref, elem.normal), yuk), (155 - s15)/2400, val
    )
    val * 2 * elem.area
end
