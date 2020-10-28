@inline function _degenerate(h::T, sinφ1::T, sinφ2::T) where T
    CUDA.max(zero(h), h) < _etol(h) ||
    1 - _abs(sinφ1) < _etol(h) ||
    1 - _abs(sinφ2) < _etol(h) ||
    _abs(sinφ1 - sinφ2) < _etol(h)
end

@inline function _distance(ξ::CuPosition{T}, n::CuPosition{T}, dist::T) where T
    _dot(ξ, n) - dist
end

@inline function _logterm(χ2::T, sinφ::T) where T
    t1 = _sqrt(one(χ2) - χ2 * sinφ * sinφ)
    t2 = _sqrt(one(χ2) - χ2) * sinφ
    (t1 + t2) / (t1 - t2)
end

@inline function _project(ξ::CuPosition{T}, n::CuPosition{T}, dist::T) where T
    _sub(ξ, _smul(n, dist))
end

@inline function _laplacepot(
    ξ   ::CuPosition{T},
    elem::CuTriangle{T},
    dist    ::T,
    pot     ::F
) where {T, F <: Function}
    _laplacepot(ξ, elem.v1, elem.v2, elem.normal, dist, pot) +
    _laplacepot(ξ, elem.v2, elem.v3, elem.normal, dist, pot) +
    _laplacepot(ξ, elem.v3, elem.v1, elem.normal, dist, pot)
end

function _laplacepot(
    ξ   ::CuPosition{T},
    x1  ::CuPosition{T},
    x2  ::CuPosition{T},
    n   ::CuPosition{T},
    dist::T,
    pot ::F
) where {T, F <: Function}
    u1  = _sub(x1, ξ)
    u1n = _norm(u1)

    u2  = _sub(x2, ξ)
    u2n = _norm(u2)

    v  = _sub(u2, u1)
    vn = _norm(v)

    sinφ1 = _clamp(_cos(u1, u1n, v, vn), -one(vn), one(vn))
    sinφ2 = _clamp(_cos(u2, u2n, v, vn), -one(vn), one(vn))

    h = _cathethus(u1n, sinφ1)

    _degenerate(h, sinφ1, sinφ2) ?
        zero(h) :
        _sign(_dot(_cross(u1, u2), n)) * pot(sinφ1, sinφ2, h, dist)
end

@inline function _laplacepot_single_plane(sinφ1::T, sinφ2::T, h::T, ::T) where T
    2 \ h * _log((1+sinφ2) * (1-sinφ1) / ((1-sinφ2) * (1+sinφ1)))
end

@inline function _laplacepot_single_space(sinφ1::T, sinφ2::T, h::T, d::T) where T
    χ2 = d * d / (d * d + h * h)
    χ  = _sqrt(χ2)
    2 \ h * _log(_logterm(χ2, sinφ2) / _logterm(χ2, sinφ1)) +
        _abs(d) * (_asin(χ * sinφ2) - _asin(sinφ2) - _asin(χ * sinφ1) + _asin(sinφ1))
end

@inline function _laplacepot_double_space(sinφ1::T, sinφ2::T, h::T, d::T) where T
    χ = _abs(d) / _sqrt(d * d + h * h)
    _sign(d) * (_asin(χ * sinφ1) - _asin(sinφ1) - _asin(χ * sinφ2) + _asin(sinφ2))
end

@inline function laplacepot_single(
    ξ   ::CuPosition{T},
    elem::CuTriangle{T}
) where T
    dist = _distance(ξ, elem.normal, elem.distorig)
    _abs(dist) < _etol(dist) ?
        _laplacepot(ξ, elem, dist, _laplacepot_single_plane) :
        _laplacepot(_project(ξ, elem.normal, dist), elem, dist, _laplacepot_single_space)
end

@inline function laplacepot_double(
    ξ   ::CuPosition{T},
    elem::CuTriangle{T}
) where T
    dist = _distance(ξ, elem.normal, elem.distorig)
    _abs(dist) < _etol(dist) ?
        zero(dist) :
        _laplacepot(_project(ξ, elem.normal, dist), elem, dist, _laplacepot_double_space)
end
