function _laplace_single_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    numξ    ::Int,
    eidx    ::Int
) where T
    for ξidx in 1:numξ
        dst[ξidx] = CuNESSie.laplacepot_single(Ξ, elements, (ξidx-1) * 3 + 1, eidx)
    end
    nothing
end

function _laplace_double_kernel!(
    dst     ::CuDeviceVector{T},
    Ξ       ::CuDeviceVector{T},
    elements::CuDeviceVector{T},
    numξ    ::Int,
    eidx    ::Int
) where T
    for ξidx in 1:numξ
        dst[ξidx] = CuNESSie.laplacepot_double(Ξ, elements, (ξidx-1) * 3 + 1, eidx)
    end
    nothing
end

@testset "[sl] ξ is vertex" begin
    for T in [Float32, Float64]
        elem = Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1])
        elements = _elem2cuarr(elem)
        Ξ = CuArray([elem.v1; elem.v2; elem.v3])
        dst = CuArray{T}(undef, 3)

        @cuda _laplace_single_kernel!(dst, Ξ, elements, 3, 1)
        res = Array(dst)
        @test res[1] ≈ Rjasanow.laplacepot(SingleLayer, elem.v1, elem, zero(T))
        @test res[2] ≈ Rjasanow.laplacepot(SingleLayer, elem.v2, elem, zero(T))
        @test res[3] ≈ Rjasanow.laplacepot(SingleLayer, elem.v3, elem, zero(T))
    end
end

@testset "[sl] ξ in plane" begin
    for T in [Float32, Float64]
        elem = Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1])
        elements = _elem2cuarr(elem)
        Ξ = CuArray(T[0, -1, 0, 0, -1, -1])
        dst = CuArray{T}(undef, 2)

        @cuda _laplace_single_kernel!(dst, Ξ, elements, 2, 1)
        res = Array(dst)
        @test res[1] ≈ Rjasanow.laplacepot(SingleLayer, T[0, -1, 0], elem, zero(T))
        @test res[2] ≈ Rjasanow.laplacepot(SingleLayer, T[0, -1, -1], elem, zero(T))
    end
end

@testset "[sl] ξ on normal through vertex" begin
    for T in [Float32, Float64]
        elem = Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1])
        elements = _elem2cuarr(elem)
        Ξ = CuArray(T[1, 0, 0, 1, 1, 0, 1, 0, 1])
        dst = CuArray{T}(undef, 3)

        @cuda _laplace_single_kernel!(dst, Ξ, elements, 3, 1)
        res = Array(dst)
        @test res[1] ≈ Rjasanow.laplacepot(SingleLayer, elem.v1, elem, one(T))
        @test res[2] ≈ Rjasanow.laplacepot(SingleLayer, elem.v2, elem, one(T))
        @test res[3] ≈ Rjasanow.laplacepot(SingleLayer, elem.v3, elem, one(T))
    end
end

@testset "[sl] ξ somewhere in space" begin
    for T in [Float32, Float64]
        elem = Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1])
        elements = _elem2cuarr(elem)
        Ξ = CuArray(T[1, -1, 0, 1, -1, -1])
        dst = CuArray{T}(undef, 2)

        @cuda _laplace_single_kernel!(dst, Ξ, elements, 2, 1)
        res = Array(dst)
        @test res[1] ≈ Rjasanow.laplacepot(SingleLayer, T[0, -1, 0], elem, one(T))
        @test res[2] ≈ Rjasanow.laplacepot(SingleLayer, T[0, -1, -1], elem, one(T))
    end
end

@testset "[dl] ξ is vertex" begin
    for T in [Float32, Float64]
        elem = Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1])
        elements = _elem2cuarr(elem)
        Ξ = CuArray([elem.v1; elem.v2; elem.v3])
        dst = CuArray{T}(undef, 3)

        @cuda _laplace_double_kernel!(dst, Ξ, elements, 3, 1)
        res = Array(dst)
        @test res[1] ≈ Rjasanow.laplacepot(DoubleLayer, elem.v1, elem, zero(T))
        @test res[2] ≈ Rjasanow.laplacepot(DoubleLayer, elem.v2, elem, zero(T))
        @test res[3] ≈ Rjasanow.laplacepot(DoubleLayer, elem.v3, elem, zero(T))
    end
end

@testset "[dl] ξ in plane" begin
    for T in [Float32, Float64]
        elem = Triangle(T[0, 0, 0], T[0, 1, 0], T[0, 0, 1])
        elements = _elem2cuarr(elem)
        Ξ = CuArray(T[0, -1, 0, 0, -1, -1])
        dst = CuArray{T}(undef, 2)

        @cuda _laplace_double_kernel!(dst, Ξ, elements, 2, 1)
        res = Array(dst)
        @test res[1] ≈ Rjasanow.laplacepot(DoubleLayer, T[0, -1, 0], elem, zero(T))
        @test res[2] ≈ Rjasanow.laplacepot(DoubleLayer, T[0, -1, -1], elem, zero(T))
    end
end

@test_skip dl_ξ_on_normal_through_vertex
@test_skip dl_ξ_somewhere_in_space
