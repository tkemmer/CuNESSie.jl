using CuNESSie: _project2d, _regularyukawapot_double, _regularyukawapot_single

@testset "_project2d" begin
    function  _project2d_kernel!(
        dst     ::CuDeviceVector{T},
        elements::CuDeviceVector{T},
        refs    ::CuDeviceVector{T},
        eidx    ::Int,
        numref  ::Int
    ) where T
        for ridx in 1:numref
            pr = _project2d(
                CuTriangle(elements, eidx),
                refs[(ridx-1) * 2 + 1],
                refs[(ridx-1) * 2 + 2]
            )
            dst[(ridx-1) * 3 + 1] = pr[1]
            dst[(ridx-1) * 3 + 2] = pr[2]
            dst[(ridx-1) * 3 + 3] = pr[3]
        end
        nothing
    end

    for T in [Float32, Float64]
        elements = elements2device([
            Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1]),
            Triangle(T[0, 0, 0], T[0, 2, 0], T[0, 0, 2]),
            Triangle(T[1, 0, 0], T[1, 2, 0], T[1, 0, 2])
        ])
        refs = CuArray(T[0, 0, 0, .5, .5, 0, .5, .5, 0, 1, 1, 0])
        dst = CuArray{T}(undef, 3 * 6)

        @cuda _project2d_kernel!(dst, elements, refs, 1, 6)
        res = Array(dst)
        @test res[1:3] ≈ T[0, 0, 0]
        @test res[4:6] ≈ T[0, 0, .5]
        @test res[7:9] ≈ T[0, .5, 0]
        @test res[10:12] ≈ T[0, .5, .5]
        @test res[13:15] ≈ T[0, 0, 1]
        @test res[16:18] ≈ T[0, 1, 0]

        @cuda _project2d_kernel!(dst, elements, refs, 15, 6)
        res = Array(dst)
        @test res[1:3] ≈ T[0, 0, 0]
        @test res[4:6] ≈ T[0, 0, 1]
        @test res[7:9] ≈ T[0, 1, 0]
        @test res[10:12] ≈ T[0, 1, 1]
        @test res[13:15] ≈ T[0, 0, 2]
        @test res[16:18] ≈ T[0, 2, 0]

        @cuda _project2d_kernel!(dst, elements, refs, 29, 6)
        res = Array(dst)
        @test res[1:3] ≈ T[1, 0, 0]
        @test res[4:6] ≈ T[1, 0, 1]
        @test res[7:9] ≈ T[1, 1, 0]
        @test res[10:12] ≈ T[1, 1, 1]
        @test res[13:15] ≈ T[1, 0, 2]
        @test res[16:18] ≈ T[1, 2, 0]
    end
end

@testset "_regularyukawapot_single" begin
    function _regularyukawapot_single_kernel!(
        dst  ::CuDeviceVector{T},
        vals ::CuDeviceVector{T},
        nvals::Int,
        yuk  ::T
    ) where T
        for i in 1:nvals
            dst[i] = _regularyukawapot_single(vals[i], yuk)
        end
        nothing
    end

    for T in [Float32, Float64]
        x    = ones(T, 3)
        o    = zeros(T, 3)
        yuk  = T(7)
        vals = CuArray(T[0, √12, √3, √(3 * .001^2), √(3 * .0001^2), √(3 * .00001^2)])
        dst  = CuArray{T}(undef, 6)

        @cuda _regularyukawapot_single_kernel!(dst, vals, 6, yuk)
        res = Array(dst)
        @test res[1] ≈ Radon.regularyukawapot(x, x, yuk)
        @test res[2] ≈ Radon.regularyukawapot(x, .-x, yuk)
        @test res[3] ≈ Radon.regularyukawapot(x, o, yuk)
        @test res[4] ≈ Radon.regularyukawapot(T(.001) .* x, o, yuk)
        @test res[5] ≈ Radon.regularyukawapot(T(.0001) .* x, o, yuk)
        @test res[6] ≈ Radon.regularyukawapot(T(.00001) .* x, o, yuk)
    end
end

@testset "_regularyukawapot_double" begin
    function _regularyukawapot_double_kernel!(
        dst  ::CuDeviceVector{T},
        vals ::CuDeviceVector{T},
        coss ::CuDeviceVector{T},
        nvals::Int,
        yuk  ::T
    ) where T
        for i in 1:nvals
            dst[i] = _regularyukawapot_double(vals[i], coss[i], yuk)
        end
        nothing
    end

    for T in [Float32, Float64]
        x    = ones(T, 3)
        o    = zeros(T, 3)
        n    = T[1, 0, 0]
        yuk  = T(7)
        vals = CuArray(T[0, √12, √3, √(3 * .001^2), √(3 * .0001^2), √(3 * .00001^2)])
        coss = CuArray(T[0, 2, 1, .001, .0001, .00001])
        dst  = CuArray{T}(undef, 6)

        @cuda _regularyukawapot_double_kernel!(dst, vals, coss, 6, yuk)
        res = Array(dst)
        @test res[1] ≈ Radon.∂ₙregularyukawapot(x, x, yuk, n)
        @test res[2] ≈ Radon.∂ₙregularyukawapot(x, .-x, yuk, n)
        @test res[3] ≈ Radon.∂ₙregularyukawapot(x, o, yuk, n)
        @test res[4] ≈ Radon.∂ₙregularyukawapot(T(.001) .* x, o, yuk, n)
        @test res[5] ≈ Radon.∂ₙregularyukawapot(T(.0001) .* x, o, yuk, n)
        @test res[6] ≈ Radon.∂ₙregularyukawapot(T(.00001) .* x, o, yuk, n)
    end
end

@testset "regularyukawapot_single" begin
    function _regularyukawapot_single_kernel!(
        dst      ::CuDeviceVector{T},
        Ξ        ::CuDeviceVector{T},
        elements ::CuDeviceVector{T},
        numξ     ::Int,
        eidx     ::Int,
        yuk      ::T
    ) where T
        elem = CuTriangle(elements, eidx)
        for ξidx in 1:numξ
            dst[ξidx] = regularyukawapot_single(CuPosition(Ξ, ξidx, numξ), elem, yuk)
        end
        nothing
    end

    for T in [Float32, Float64]
        yuk = T(7)
        Ξ = Ξ2device([
            T[0, 0, 0],  T[0, 1, 0],   T[0, 0, 1],
            T[0, -1, 0], T[0, -1, -1],
            T[1, 0, 0],  T[1, 1, 0],   T[1, 0, 1],
            T[1, -1, 0], T[1, -1, -1]
        ])
        dst  = CuArray{T}(undef, length(Ξ))

        for elem in [
            Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1]),
            Triangle(T[0, 0, 0], T[0, 2, 0], T[0, 0, 2]),
            Triangle(T[1, 0, 0], T[1, 2, 0], T[1, 0, 2])
        ]
            @cuda _regularyukawapot_single_kernel!(
                dst, Ξ, elements2device([elem]), 10, 1, yuk)
            res = Array(dst)
            @test res[1] ≈ Radon.regularyukawacoll(SingleLayer, T[0, 0, 0], elem, yuk)
            @test res[2] ≈ Radon.regularyukawacoll(SingleLayer, T[0, 1, 0], elem, yuk)
            @test res[3] ≈ Radon.regularyukawacoll(SingleLayer, T[0, 0, 1], elem, yuk)
            @test res[4] ≈ Radon.regularyukawacoll(SingleLayer, T[0, -1, 0], elem, yuk)
            @test res[5] ≈ Radon.regularyukawacoll(SingleLayer, T[0, -1, -1], elem, yuk)
            @test res[6] ≈ Radon.regularyukawacoll(SingleLayer, T[1, 0, 0], elem, yuk)
            @test res[7] ≈ Radon.regularyukawacoll(SingleLayer, T[1, 1, 0], elem, yuk)
            @test res[8] ≈ Radon.regularyukawacoll(SingleLayer, T[1, 0, 1], elem, yuk)
            @test res[9] ≈ Radon.regularyukawacoll(SingleLayer, T[1, -1, 0], elem, yuk)
            @test res[10] ≈ Radon.regularyukawacoll(SingleLayer, T[1, -1, -1], elem, yuk)
        end
    end
end

@testset "regularyukawapot_double" begin
    function _regularyukawapot_double_kernel!(
        dst      ::CuDeviceVector{T},
        Ξ        ::CuDeviceVector{T},
        elements ::CuDeviceVector{T},
        numξ     ::Int,
        eidx     ::Int,
        yuk      ::T
    ) where T
        elem = CuTriangle(elements, eidx)
        for ξidx in 1:numξ
            dst[ξidx] = regularyukawapot_double(CuPosition(Ξ, ξidx, numξ), elem, yuk)
        end
        nothing
    end

    for T in [Float32, Float64]
        yuk = T(7)
        Ξ = Ξ2device([
            T[0, 0, 0],  T[0, 1, 0],   T[0, 0, 1],
            T[0, -1, 0], T[0, -1, -1],
            T[1, 0, 0],  T[1, 1, 0],   T[1, 0, 1],
            T[1, -1, 0], T[1, -1, -1]
        ])
        dst  = CuArray{T}(undef, length(Ξ))

        for elem in [
            Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1]),
            Triangle(T[0, 0, 0], T[0, 2, 0], T[0, 0, 2]),
            Triangle(T[1, 0, 0], T[1, 2, 0], T[1, 0, 2])
        ]
            @cuda _regularyukawapot_double_kernel!(
                dst, Ξ, elements2device([elem]), 10, 1, yuk)
            res = Array(dst)
            @test res[1] ≈ Radon.regularyukawacoll(DoubleLayer, T[0, 0, 0], elem, yuk)
            @test res[2] ≈ Radon.regularyukawacoll(DoubleLayer, T[0, 1, 0], elem, yuk)
            @test res[3] ≈ Radon.regularyukawacoll(DoubleLayer, T[0, 0, 1], elem, yuk)
            @test res[4] ≈ Radon.regularyukawacoll(DoubleLayer, T[0, -1, 0], elem, yuk)
            @test res[5] ≈ Radon.regularyukawacoll(DoubleLayer, T[0, -1, -1], elem, yuk)
            @test res[6] ≈ Radon.regularyukawacoll(DoubleLayer, T[1, 0, 0], elem, yuk)
            @test res[7] ≈ Radon.regularyukawacoll(DoubleLayer, T[1, 1, 0], elem, yuk)
            @test res[8] ≈ Radon.regularyukawacoll(DoubleLayer, T[1, 0, 1], elem, yuk)
            @test res[9] ≈ Radon.regularyukawacoll(DoubleLayer, T[1, -1, 0], elem, yuk)
            @test res[10] ≈ Radon.regularyukawacoll(DoubleLayer, T[1, -1, -1], elem, yuk)
        end
    end
end
