@inline function _project2d(
    elements::CuDeviceVector{T},
    eidx    ::Int,
    rx      ::T,
    ry      ::T,
) where T
    v1x, v1y, v1z = _pos(elements, eidx)
    (
        CUDAnative.fma(
            elements[eidx+3]-v1x,
            rx,
            CUDAnative.fma(elements[eidx+6]-v1x, ry, v1x)
        ),
        CUDAnative.fma(
            elements[eidx+4]-v1y,
            rx,
            CUDAnative.fma(elements[eidx+7]-v1y, ry, v1y)
        ),
        CUDAnative.fma(
            elements[eidx+5]-v1z,
            rx,
            CUDAnative.fma(elements[eidx+8]-v1z, ry, v1z)
        )
    )
end

@inline function _refside(
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    ξidx    ::Int,
    eidx    ::Int,
    rx      ::T,
    ry      ::T
) where T
    _sub(_project2d(elements, eidx, rx, ry)..., _pos(Ξ, ξidx)...)
end

@inline function _regularyukawapot_single(rn ::T, yuk::T) where T
    sn = yuk * rn
    rn < 1e-10 ?
        -yuk : sn >= 0.1 ?
        CUDAnative.expm1(-sn) / rn :
        _regularyukawapot_single_series(sn) / rn
end

function _regularyukawapot_single_series(sn::T) where T
    term = -sn
    tol  = 1e-10 * _abs(term)
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

    rn < 1e-10 ?
        zero(rn) : sn >= 0.1 ?
        (one(rn) - (one(rn) + sn) * CUDAnative.exp(-sn)) * cs / rn / rn / rn :
        _regularyukawapot_double_series(sn) * cs / rn / rn / rn
end

function _regularyukawapot_double_series(sn::T) where T
    term = 2 \ sn * sn
    tol  = 1e-10 * _abs(term)
    tsum = zero(sn)
    for i in 2:16
        _abs(term) <= tol && break
        tsum = CUDAnative.fma(term, T(i-1), tsum)
        term *= -sn / (i+1)
    end
    tsum
end

function regularyukawapot_single(
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    ξidx    ::Int,
    eidx    ::Int,
    yuk     ::T
) where T
    unit = one(yuk)
    s15  = _sqrt(unit * 15)
    pot  = _regularyukawapot_single

    val = zero(yuk)
    val = CUDAnative.fma(
        pot(_norm(_refside(Ξ, elements, ξidx, eidx, unit/3, unit/3)...), yuk),
        unit * 9 / 80,
        val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(Ξ, elements, ξidx, eidx, (6+s15)/21, (9-2*s15)/21)...), yuk),
        (155 + s15)/2400,
        val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(Ξ, elements, ξidx, eidx, (9-2*s15)/21, (6+s15)/21)...), yuk),
        (155 + s15)/2400,
        val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(Ξ, elements, ξidx, eidx, (6+s15)/21, (6+s15)/21)...), yuk),
        (155 + s15)/2400,
        val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(Ξ, elements, ξidx, eidx, (6-s15)/21, (9+2*s15)/21)...), yuk),
        (155 - s15)/2400,
        val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(Ξ, elements, ξidx, eidx, (9+2*s15)/21, (6-s15)/21)...), yuk),
        (155 - s15)/2400,
        val
    )
    val = CUDAnative.fma(
        pot(_norm(_refside(Ξ, elements, ξidx, eidx, (6-s15)/21, (6-s15)/21)...), yuk),
        (155 - s15)/2400,
        val
    )
    val * 2 * elements[eidx + 13]
end

function regularyukawapot_double(
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    ξidx    ::Int,
    eidx    ::Int,
    yuk     ::T
) where T
    unit = one(yuk)
    s15  = _sqrt(unit * 15)
    pot  = _regularyukawapot_double

    val = zero(yuk)
    ref = _refside(Ξ, elements, ξidx, eidx, unit/3, unit/3)
    val = CUDAnative.fma(
        pot(
            _norm(ref...),
            _dot(ref..., _pos(elements, eidx + 9)...),
            yuk
        ),
        unit * 9 / 80,
        val
    )
    ref = _refside(Ξ, elements, ξidx, eidx, (6+s15)/21, (9-2*s15)/21)
    val = CUDAnative.fma(
        pot(
            _norm(ref...),
            _dot(ref..., _pos(elements, eidx + 9)...),
            yuk
        ),
        (155 + s15)/2400,
        val
    )
    ref = _refside(Ξ, elements, ξidx, eidx, (9-2*s15)/21, (6+s15)/21)
    val = CUDAnative.fma(
        pot(
            _norm(ref...),
            _dot(ref..., _pos(elements, eidx + 9)...),
            yuk
        ),
        (155 + s15)/2400,
        val
    )
    ref = _refside(Ξ, elements, ξidx, eidx, (6+s15)/21, (6+s15)/21)
    val = CUDAnative.fma(
        pot(
            _norm(ref...),
            _dot(ref..., _pos(elements, eidx + 9)...),
            yuk
        ),
        (155 + s15)/2400,
        val
    )
    ref = _refside(Ξ, elements, ξidx, eidx, (6-s15)/21, (9+2*s15)/21)
    val = CUDAnative.fma(
        pot(
            _norm(ref...),
            _dot(ref..., _pos(elements, eidx + 9)...),
            yuk
        ),
        (155 - s15)/2400,
        val
    )
    ref = _refside(Ξ, elements, ξidx, eidx, (9+2*s15)/21, (6-s15)/21)
    val = CUDAnative.fma(
        pot(
            _norm(ref...),
            _dot(ref..., _pos(elements, eidx + 9)...),
            yuk
        ),
        (155 - s15)/2400,
        val
    )
    ref = _refside(Ξ, elements, ξidx, eidx, (6-s15)/21, (6-s15)/21)
    val = CUDAnative.fma(
        pot(
            _norm(ref...),
            _dot(ref..., _pos(elements, eidx + 9)...),
            yuk
        ),
        (155 - s15)/2400,
        val
    )
    val * 2 * elements[eidx + 13]
end
