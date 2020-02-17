@inline function _abs(x::T) where T
    CUDAnative.abs(x)
end

@inline function _asin(x::T) where T
    CUDAnative.asin(x)
end

@inline function _cathethus(hyp::T, cos_theta::T) where T
    CUDAnative.sqrt(hyp * hyp * (1 - cos_theta * cos_theta))
end

@inline function _clamp(x::T, lo::T, hi::T) where T
    CUDAnative.max(lo, CUDAnative.min(hi, x))
end

# deprecated
@inline function _cos(x1::T, y1::T, z1::T, n1::T, x2::T, y2::T, z2::T, n2::T) where T
    _dot(x1, y1, z1, x2, y2, z2) / (n1 * n2)
end

@inline function _cos(u::CuPosition{T}, unorm::T, v::CuPosition{T}, vnorm::T) where T
    _dot(u, v) / (unorm * vnorm)
end

# deprecated
@inline function _cross(x1::T, y1::T, z1::T, x2::T, y2::T, z2::T) where T
    (y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2)
end

@inline function _cross(u::CuPosition{T}, v::CuPosition{T}) where T
    (
        x = u.y * v.z - u.z * v.y,
        y = u.z * v.x - u.x * v.z,
        z = u.x * v.y - u.y * v.x
    )
end

# deprecated
@inline function _dot(x1::T, y1::T, z1::T, x2::T, y2::T, z2::T) where T
    CUDAnative.fma(x1, x2, CUDAnative.fma(y1, y2, z1 * z2))
end

@inline function _dot(u::CuPosition{T}, v::CuPosition{T}) where T
    CUDAnative.fma(u.x, v.x, CUDAnative.fma(u.y, v.y, u.z * v.z))
end

@inline function _log(x::T) where T
    CUDAnative.log(x)
end

# deprecated
@inline function _norm(x::T, y::T, z::T) where T
    CUDAnative.sqrt(_dot(x, y, z, x, y, z))
end

@inline function _norm(v::CuPosition{T}) where T
    CUDAnative.sqrt(_dot(v, v))
end

# deprecated
@inline function _pos(x::CuDeviceVector{T}, idx::Int) where T
    (x[idx], x[idx + 1], x[idx + 2])
end

@inline function _sign(x::T) where T
    x == 0 ? zero(x) : x / _abs(x)
end

@inline function _sqrt(x::T) where T
    CUDAnative.sqrt(x)
end

@inline function _smul(v::CuPosition{T}, s::T) where T
    (x = s * v.x, y = s * v.y, z = s * v.z)
end

# deprecated
@inline function _sub(x1::T, y1::T, z1::T, x2::T, y2::T, z2::T) where T
    (x1 - x2, y1 - y2, z1 - z2)
end

@inline function _sub(u::CuPosition{T}, v::CuPosition{T}) where T
    (x = u.x - v.x, y = u.y - v.y, z = u.z - v.z)
end
