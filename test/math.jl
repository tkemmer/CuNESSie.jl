using CuNESSie: CuPosition, _clamp, _cos, _cross, _dot, _norm, _sign, _smul, _sub
using LinearAlgebra: ×

@test_skip "_cathethus"

@testset "_clamp" begin
    function _clamp_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}) where T
        for i in 1:9
            dst[i] = _clamp(vals[i+2], vals[1], vals[2])
        end
        nothing
    end

    for T in [Float32, Float64]
        vals = CuArray(T[-1, 1, -2, -1.0001, -1, -.999, 0, .999, 1, 1.0001, 2])
        dst  = CuArray{T}(undef, 9)

        @cuda _clamp_kernel!(dst, vals)
        res = Array(dst)
        @test res ≈ T[-1, -1, -1, -.999, 0, .999, 1, 1, 1]
    end
end

@testset "_cos" begin
    function _cos_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}, n::Int) where T
        u = CuPosition(vals, 1, n)
        un = _norm(u)
        dst[1] = _cos(u, un, u, un)
        for i in 0:5
            v = CuPosition(vals, i + 2, n)
            vn = _norm(v)
            dst[2i + 2] = _cos(u, un, v, vn)
            dst[2i + 3] = _cos(v, vn, u, un)
        end
        nothing
    end

    for T in [Float32, Float64]
        vals = _pos2cuxi(
            T[1, 0, 0],
            T[0, 1, 0], T[-1, 0, 0], T[0, -1, 0],
            T[1, 1, 0], T[-1, 1, 0], T[1, 0, 1]
        )
        dst  = CuArray{T}(undef, 13)

        @cuda _cos_kernel!(dst, vals, 7)
        res = Array(dst)
        @test res[1] ≈ one(T)
        @test res[2] ≈ zero(T)
        @test res[3] ≈ zero(T)
        @test res[4] ≈ -one(T)
        @test res[5] ≈ -one(T)
        @test res[6] ≈ zero(T)
        @test res[7] ≈ zero(T)
        @test res[8] ≈ T(cos(π/4))
        @test res[9] ≈ T(cos(π/4))
        @test res[10] ≈ T(cos(3π/4))
        @test res[11] ≈ T(cos(3π/4))
        @test res[12] ≈ T(cos(π/4))
        @test res[13] ≈ T(cos(π/4))
    end
end

@testset "_cross" begin
    function _cross_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}, n::Int) where T
        v1 = CuPosition(vals, 1, n)
        v2 = CuPosition(vals, 2, n)
        v3 = CuPosition(vals, 3, n)
        v4 = CuPosition(vals, 4, n)
        dst[1], dst[2], dst[3] = _cross(v1, v2)
        dst[4], dst[5], dst[6] = _cross(v2, v1)
        dst[7], dst[8], dst[9] = _cross(v1, v3)
        dst[10], dst[11], dst[12] = _cross(v3, v1)
        dst[13], dst[14], dst[15] = _cross(v1, v4)
        dst[16], dst[17], dst[18] = _cross(v4, v1)
        nothing
    end

    for T in [Float32, Float64]
        vals = _pos2cuxi(T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0])
        dst  = CuArray{T}(undef, 3 * 6)

        @cuda _cross_kernel!(dst, vals, 4)
        res = Array(dst)
        @test res[1:3] ≈ T[1, 2, 3] × T[4, 6, 8]
        @test res[4:6] ≈ T[4, 6, 8] × T[1, 2, 3]
        @test res[7:9] ≈ T[1, 2, 3] × T[-1, -1, -1]
        @test res[10:12] ≈ T[-1, -1, -1] × T[1, 2, 3]
        @test res[13:15] ≈ T[1, 2, 3] × T[0, 0, 0]
        @test res[16:18] ≈ T[0, 0, 0] × T[1, 2, 3]
    end
end

@testset "_dot" begin
    function _dot_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}, n::Int) where T
        v1 = CuPosition(vals, 1, n)
        v2 = CuPosition(vals, 2, n)
        v3 = CuPosition(vals, 3, n)
        v4 = CuPosition(vals, 4, n)
        dst[1] = _dot(v1, v2)
        dst[2] = _dot(v2, v1)
        dst[3] = _dot(v1, v3)
        dst[4] = _dot(v3, v1)
        dst[5] = _dot(v1, v4)
        dst[6] = _dot(v4, v1)
        nothing
    end

    for T in [Float32, Float64]
        vals = _pos2cuxi(T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0])
        dst  = CuArray{T}(undef, 6)

        @cuda _dot_kernel!(dst, vals, 4)
        res = Array(dst)
        @test res ≈ T[40, 40, -6, -6, 0, 0]
    end
end

@testset "_norm" begin
    function _norm_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}, n::Int) where T
        dst[1] = _norm(CuPosition(vals, 1, n))
        dst[2] = _norm(CuPosition(vals, 2, n))
        dst[3] = _norm(CuPosition(vals, 3, n))
        dst[4] = _norm(CuPosition(vals, 4, n))
        nothing
    end

    for T in [Float32, Float64]
        vals = _pos2cuxi(T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0])
        dst  = CuArray{T}(undef, 4)

        @cuda _norm_kernel!(dst, vals, 4)
        res = Array(dst)
        @test res ≈ T[√14, √116, √3, 0]
    end
end

@testset "_sign" begin
    function _sign_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}) where T
        for i in 1:6
            dst[i] = _sign(vals[i])
        end
        nothing
    end

    for T in [Float32, Float64]
        vals = CuArray(T[0, -0, 1, 2, -1, -2])
        dst  = CuArray{T}(undef, 6)

        @cuda _sign_kernel!(dst, vals)
        res = Array(dst)
        @test res ≈ T[0, 0, 1, 1, -1, -1]
    end
end

@testset "_smul" begin
    function _smul_kernel!(
        dst::CuDeviceVector{T},
        vals::CuDeviceVector{T},
        s::T,
        n::Int
    ) where T
        dst[1], dst[2], dst[3] = _smul(CuPosition(vals, 1, n), s)
        dst[4], dst[5], dst[6] = _smul(CuPosition(vals, 2, n), s)
        dst[7], dst[8], dst[9] = _smul(CuPosition(vals, 3, n), s)
        dst[10], dst[11], dst[12] = _smul(CuPosition(vals, 4, n), s)
        nothing
    end

    for T in [Float32, Float64]
        vals = _pos2cuxi(T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0])
        dst  = CuArray{T}(undef, 12)

        @cuda _smul_kernel!(dst, vals, T(2), 4)
        res = Array(dst)
        @test res[1:3] ≈ T[2, 4, 6]
        @test res[4:6] ≈ T[8, 12, 16]
        @test res[7:9] ≈ -T[2, 2, 2]
        @test res[10:12] ≈ T[0, 0, 0]

        @cuda _smul_kernel!(dst, vals, T(-2), 4)
        res = Array(dst)
        @test res[1:3] ≈ -T[2, 4, 6]
        @test res[4:6] ≈ -T[8, 12, 16]
        @test res[7:9] ≈ T[2, 2, 2]
        @test res[10:12] ≈ -T[0, 0, 0]
    end
end

@testset "_sub" begin
    function _sub_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}, n::Int) where T
        v1 = CuPosition(vals, 1, n)
        v2 = CuPosition(vals, 2, n)
        v3 = CuPosition(vals, 3, n)
        v4 = CuPosition(vals, 4, n)
        dst[1], dst[2], dst[3] = _sub(v1, v2)
        dst[4], dst[5], dst[6] = _sub(v2, v1)
        dst[7], dst[8], dst[9] = _sub(v1, v3)
        dst[10], dst[11], dst[12] = _sub(v3, v1)
        dst[13], dst[14], dst[15] = _sub(v1, v4)
        dst[16], dst[17], dst[18] = _sub(v4, v1)
        nothing
    end

    for T in [Float32, Float64]
        vals = _pos2cuxi(T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0])
        dst  = CuArray{T}(undef, 3 * 6)

        @cuda _sub_kernel!(dst, vals, 4)
        res = Array(dst)
        @test res[1:3] ≈ T[-3, -4, -5]
        @test res[4:6] ≈ T[3, 4, 5]
        @test res[7:9] ≈ T[2, 3, 4]
        @test res[10:12] ≈ T[-2, -3, -4]
        @test res[13:15] ≈ T[1, 2, 3]
        @test res[16:18] ≈ T[-1, -2, -3]
    end
end
