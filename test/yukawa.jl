function  _project2d_kernel!(
    dst     ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    refs    ::CuDeviceVector{T},
    eidx    ::Int,
    numref  ::Int
) where T
    for ridx in 1:numref
        pr = CuNESSie._project2d(
            elements,
            eidx,
            refs[(ridx-1) * 2 + 1],
            refs[(ridx-1) * 2 + 2]
        )
        dst[(ridx-1) * 3 + 1] = pr[1]
        dst[(ridx-1) * 3 + 2] = pr[2]
        dst[(ridx-1) * 3 + 3] = pr[3]
    end
    nothing
end

@testset "2D projection" begin
    for T in [Float32, Float64]
        elements = _elem2cuarr(
            Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1]),
            Triangle(T[0, 0, 0], T[0, 2, 0], T[0, 0, 2]),
            Triangle(T[1, 0, 0], T[1, 2, 0], T[1, 0, 2])
        )
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

@testset "single layer" begin
    for T in [Float32, Float64]

    end
end
