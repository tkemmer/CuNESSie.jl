@inline function _degenerate(h::T, sinφ1::T, sinφ2::T) where T
    CUDAnative.max(zero(h), h) < 1e-5 ||
    1 - _abs(sinφ1) < 1e-5 ||
    1 - _abs(sinφ2) < 1e-5 ||
    _abs(sinφ1 - sinφ2) < 1e-5
end

@inline function _distance(ξx::T, ξy::T, ξz::T, nx::T, ny::T, nz::T, dist::T) where T
    _dot(ξx, ξy, ξz, nx, ny, nz) - dist
end

@inline function _logterm(χ2::T, sinφ::T) where T
    t1 = _sqrt(one(χ2) - χ2 * sinφ * sinφ)
    t2 = _sqrt(one(χ2) - χ2) * sinφ
    (t1 + t2) / (t1 - t2)
end

@inline function _project(ξx::T, ξy::T, ξz::T, nx::T, ny::T, nz::T, dist::T) where T
    _sub(ξx, ξy, ξz, nx * dist, ny * dist, nz * dist)
end


@inline function _laplacepot(
    elements::CuDeviceVector{T},
    ξx      ::T,
    ξy      ::T,
    ξz      ::T,
    eidx    ::Int,
    dist    ::T,
    pot     ::F
) where {T, F <: Function}
    _laplacepot(elements, ξx, ξy, ξz, eidx,     eidx + 3, eidx + 9, dist, pot) +
    _laplacepot(elements, ξx, ξy, ξz, eidx + 3, eidx + 6, eidx + 9, dist, pot) +
    _laplacepot(elements, ξx, ξy, ξz, eidx + 6, eidx,     eidx + 9, dist, pot)
end

function _laplacepot(
    elements::CuDeviceVector{T},
    ξx      ::T,
    ξy      ::T,
    ξz      ::T,
    x1idx   ::Int,
    x2idx   ::Int,
    nidx    ::Int,
    dist    ::T,
    pot     ::F
) where {T, F <: Function}
    u1x = elements[x1idx] - ξx
    u1y = elements[x1idx + 1] - ξy
    u1z = elements[x1idx + 2] - ξz
    u1n = _norm(u1x, u1y, u1z)

    u2x = elements[x2idx] - ξx
    u2y = elements[x2idx + 1] - ξy
    u2z = elements[x2idx + 2] - ξz
    u2n = _norm(u2x, u2y, u2z)

    vx = elements[x2idx] - elements[x1idx]
    vy = elements[x2idx + 1] - elements[x1idx + 1]
    vz = elements[x2idx + 2] - elements[x1idx + 2]
    vn = _norm(vx, vy, vz)

    sinφ1 = _clamp(_cos(u1x, u1y, u1z, u1n, vx, vy, vz, vn), -one(vn), one(vn))
    sinφ2 = _clamp(_cos(u2x, u2y, u2z, u2n, vx, vy, vz, vn), -one(vn), one(vn))

    h = _cathethus(u1n, sinφ1)

    _degenerate(h, sinφ1, sinφ2) ?
        zero(h) :
        _sign(_dot(_cross(u1x, u1y, u1z, u2x, u2y, u2z)..., _pos(elements, nidx)...)) *
            pot(sinφ1, sinφ2, h, dist)
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

# (unused)
@inline function _laplacepot_double_plane(::T, ::T, h::T, ::T) where T
    zero(h)
end

@inline function _laplacepot_double_space(sinφ1::T, sinφ2::T, h::T, d::T) where T
    χ = _abs(d) / _sqrt(d * d + h * h)
    _sign(d) * (_asin(χ * sinφ1) - _asin(sinφ1) - _asin(χ * sinφ2) + _asin(sinφ2))
end

@inline function laplacepot_single(
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    ξidx    ::Int,
    eidx    ::Int
) where T
    dist = _distance(_pos(Ξ, ξidx)..., _pos(elements, eidx + 9)..., elements[eidx + 12])
    _abs(dist) < 1e-5 ?
        _laplacepot(elements, _pos(Ξ, ξidx)..., eidx, dist, _laplacepot_single_plane) :
        _laplacepot(elements,
            _project(_pos(Ξ, ξidx)..., _pos(elements, eidx + 9)..., dist)...,
            eidx, dist, _laplacepot_single_space)
end

@inline function laplacepot_double(
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    ξidx    ::Int,
    eidx    ::Int
) where T
    dist = _distance(_pos(Ξ, ξidx)..., _pos(elements, eidx + 9)..., elements[eidx + 12])
    _abs(dist) < 1e-5 ?
        zero(dist) :
        _laplacepot(elements,
            _project(_pos(Ξ, ξidx)..., _pos(elements, eidx + 9)..., dist)...,
            eidx, dist, _laplacepot_double_space)
end
