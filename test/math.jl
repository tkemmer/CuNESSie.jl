using CuNESSie: _clamp, _cos, _cross, _dot, _norm, _pos, _sign, _sub
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
    function _cos_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}) where T
        dst[1] = _cos(
            _pos(vals, 1)..., _norm(_pos(vals, 1)...),
            _pos(vals, 1)..., _norm(_pos(vals, 1)...)
        )
        for i in 0:5
            j = 1 + (i+1) * 3
            dst[2i + 2] = _cos(
                _pos(vals, 1)..., _norm(_pos(vals, 1)...),
                _pos(vals, j)..., _norm(_pos(vals, j)...)
            )
            dst[2i + 3] = _cos(
                _pos(vals, j)..., _norm(_pos(vals, j)...),
                _pos(vals, 1)..., _norm(_pos(vals, 1)...)
            )
        end
        nothing
    end

    for T in [Float32, Float64]
        vals = CuArray(T[1, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 1, 1, 0, -1, 1, 0, 1, 0, 1])
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
    function _cross_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}) where T
        dst[1], dst[2], dst[3] = _cross(_pos(vals, 1)..., _pos(vals, 4)...)
        dst[4], dst[5], dst[6] = _cross(_pos(vals, 4)..., _pos(vals, 1)...)
        dst[7], dst[8], dst[9] = _cross(_pos(vals, 1)..., _pos(vals, 7)...)
        dst[10], dst[11], dst[12] = _cross(_pos(vals, 7)..., _pos(vals, 1)...)
        dst[13], dst[14], dst[15] = _cross(_pos(vals, 1)..., _pos(vals, 10)...)
        dst[16], dst[17], dst[18] = _cross(_pos(vals, 10)..., _pos(vals, 1)...)
        nothing
    end

    for T in [Float32, Float64]
        vals = CuArray(T[1, 2, 3, 4, 6, 8, -1, -1, -1, 0, 0, 0])
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
    function _dot_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}) where T
        dst[1] = _dot(_pos(vals, 1)..., _pos(vals, 4)...)
        dst[2] = _dot(_pos(vals, 4)..., _pos(vals, 1)...)
        dst[3] = _dot(_pos(vals, 1)..., _pos(vals, 7)...)
        dst[4] = _dot(_pos(vals, 7)..., _pos(vals, 1)...)
        dst[5] = _dot(_pos(vals, 1)..., _pos(vals, 10)...)
        dst[6] = _dot(_pos(vals, 10)..., _pos(vals, 1)...)
        nothing
    end

    for T in [Float32, Float64]
        vals = CuArray(T[1, 2, 3, 4, 6, 8, -1, -1, -1, 0, 0, 0])
        dst  = CuArray{T}(undef, 6)

        @cuda _dot_kernel!(dst, vals)
        res = Array(dst)
        @test res ≈ T[40, 40, -6, -6, 0, 0]
    end
end

@testset "_norm" begin
    function _norm_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}) where T
        dst[1] = _norm(_pos(vals, 1)...)
        dst[2] = _norm(_pos(vals, 4)...)
        dst[3] = _norm(_pos(vals, 7)...)
        dst[4] = _norm(_pos(vals, 10)...)
        nothing
    end

    for T in [Float32, Float64]
        vals = CuArray(T[1, 2, 3, 4, 6, 8, -1, -1, -1, 0, 0, 0])
        dst  = CuArray{T}(undef, 4)

        @cuda _norm_kernel!(dst, vals)
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

@testset "_sub" begin
    function _sub_kernel!(dst::CuDeviceVector{T}, vals::CuDeviceVector{T}) where T
        dst[1], dst[2], dst[3] = _sub(_pos(vals, 1)..., _pos(vals, 4)...)
        dst[4], dst[5], dst[6] = _sub(_pos(vals, 4)..., _pos(vals, 1)...)
        dst[7], dst[8], dst[9] = _sub(_pos(vals, 1)..., _pos(vals, 7)...)
        dst[10], dst[11], dst[12] = _sub(_pos(vals, 7)..., _pos(vals, 1)...)
        dst[13], dst[14], dst[15] = _sub(_pos(vals, 1)..., _pos(vals, 10)...)
        dst[16], dst[17], dst[18] = _sub(_pos(vals, 10)..., _pos(vals, 1)...)
        nothing
    end

    for T in [Float32, Float64]
        vals = CuArray(T[1, 2, 3, 4, 6, 8, -1, -1, -1, 0, 0, 0])
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
