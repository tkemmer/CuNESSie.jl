@testitem "Math utilities" begin
    using CUDA

    using CuNESSie:
        CuDeviceVector,
        CuPositionVector,
        PositionVector

    @test_skip "_cathethus"

    @testset "_clamp" begin
        using CuNESSie: _clamp

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
        using CuNESSie: _cos, _norm

        function _cos_kernel!(dst::CuDeviceVector{T}, vals::CuPositionVector{T}) where T
            u = vals[1]
            un = _norm(u)
            dst[1] = _cos(u, un, u, un)
            for i in 0:5
                v = vals[i + 2]
                vn = _norm(v)
                dst[2i + 2] = _cos(u, un, v, vn)
                dst[2i + 3] = _cos(v, vn, u, un)
            end
            nothing
        end

        for T in [Float32, Float64]
            vals = PositionVector([
                T[1, 0, 0],
                T[0, 1, 0], T[-1, 0, 0], T[0, -1, 0],
                T[1, 1, 0], T[-1, 1, 0], T[1, 0, 1]
            ])
            dst  = CuArray{T}(undef, 13)

            @cuda _cos_kernel!(dst, vals)
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
        using CuNESSie: _cross
        using LinearAlgebra: ×

        function _cross_kernel!(dst::CuDeviceVector{T}, vals::CuPositionVector{T}) where T
            v1 = vals[1]
            v2 = vals[2]
            v3 = vals[3]
            v4 = vals[4]
            dst[1], dst[2], dst[3] = _cross(v1, v2)
            dst[4], dst[5], dst[6] = _cross(v2, v1)
            dst[7], dst[8], dst[9] = _cross(v1, v3)
            dst[10], dst[11], dst[12] = _cross(v3, v1)
            dst[13], dst[14], dst[15] = _cross(v1, v4)
            dst[16], dst[17], dst[18] = _cross(v4, v1)
            nothing
        end

        for T in [Float32, Float64]
            vals = PositionVector([T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0]])
            dst  = CuArray{T}(undef, 3 * 6)

            @cuda _cross_kernel!(dst, vals)
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
        using CuNESSie: _dot

        function _dot_kernel!(dst::CuDeviceVector{T}, vals::CuPositionVector{T}) where T
            v1 = vals[1]
            v2 = vals[2]
            v3 = vals[3]
            v4 = vals[4]
            dst[1] = _dot(v1, v2)
            dst[2] = _dot(v2, v1)
            dst[3] = _dot(v1, v3)
            dst[4] = _dot(v3, v1)
            dst[5] = _dot(v1, v4)
            dst[6] = _dot(v4, v1)
            nothing
        end

        for T in [Float32, Float64]
            vals = PositionVector([T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0]])
            dst  = CuArray{T}(undef, 6)

            @cuda _dot_kernel!(dst, vals)
            res = Array(dst)
            @test res ≈ T[40, 40, -6, -6, 0, 0]
        end
    end

    @testset "_norm" begin
        using CuNESSie: _norm

        function _norm_kernel!(dst::CuDeviceVector{T}, vals::CuPositionVector{T}) where T
            dst[1] = _norm(vals[1])
            dst[2] = _norm(vals[2])
            dst[3] = _norm(vals[3])
            dst[4] = _norm(vals[4])
            nothing
        end

        for T in [Float32, Float64]
            vals = PositionVector([T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0]])
            dst  = CuArray{T}(undef, 4)

            @cuda _norm_kernel!(dst, vals)
            res = Array(dst)
            @test res ≈ T[√14, √116, √3, 0]
        end
    end

    @testset "_sign" begin
        using CuNESSie: _sign

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
        using CuNESSie: _smul

        function _smul_kernel!(
            dst::CuDeviceVector{T},
            vals::CuPositionVector{T},
            s::T
        ) where T
            dst[1], dst[2], dst[3] = _smul(vals[1], s)
            dst[4], dst[5], dst[6] = _smul(vals[2], s)
            dst[7], dst[8], dst[9] = _smul(vals[3], s)
            dst[10], dst[11], dst[12] = _smul(vals[4], s)
            nothing
        end

        for T in [Float32, Float64]
            vals = PositionVector([T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0]])
            dst  = CuArray{T}(undef, 12)

            @cuda _smul_kernel!(dst, vals, T(2))
            res = Array(dst)
            @test res[1:3] ≈ T[2, 4, 6]
            @test res[4:6] ≈ T[8, 12, 16]
            @test res[7:9] ≈ -T[2, 2, 2]
            @test res[10:12] ≈ T[0, 0, 0]

            @cuda _smul_kernel!(dst, vals, T(-2))
            res = Array(dst)
            @test res[1:3] ≈ -T[2, 4, 6]
            @test res[4:6] ≈ -T[8, 12, 16]
            @test res[7:9] ≈ T[2, 2, 2]
            @test res[10:12] ≈ -T[0, 0, 0]
        end
    end

    @testset "_sub" begin
        using CuNESSie: _sub

        function _sub_kernel!(dst::CuDeviceVector{T}, vals::CuPositionVector{T}) where T
            v1 = vals[1]
            v2 = vals[2]
            v3 = vals[3]
            v4 = vals[4]
            dst[1], dst[2], dst[3] = _sub(v1, v2)
            dst[4], dst[5], dst[6] = _sub(v2, v1)
            dst[7], dst[8], dst[9] = _sub(v1, v3)
            dst[10], dst[11], dst[12] = _sub(v3, v1)
            dst[13], dst[14], dst[15] = _sub(v1, v4)
            dst[16], dst[17], dst[18] = _sub(v4, v1)
            nothing
        end

        for T in [Float32, Float64]
            vals = PositionVector([T[1, 2, 3], T[4, 6, 8], T[-1, -1, -1], T[0, 0, 0]])
            dst  = CuArray{T}(undef, 3 * 6)

            @cuda _sub_kernel!(dst, vals)
            res = Array(dst)
            @test res[1:3] ≈ T[-3, -4, -5]
            @test res[4:6] ≈ T[3, 4, 5]
            @test res[7:9] ≈ T[2, 3, 4]
            @test res[10:12] ≈ T[-2, -3, -4]
            @test res[13:15] ≈ T[1, 2, 3]
            @test res[16:18] ≈ T[-1, -2, -3]
        end
    end
end
