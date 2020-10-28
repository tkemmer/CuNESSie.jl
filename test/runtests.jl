using CUDA
using NESSie
using ImplicitArrays
using Test

using CuNESSie
using CuNESSie: CuPosition, CuPositionVector, CuTriangle, CuTriangleVector, PositionVector,
    TriangleVector, laplacepot_single, laplacepot_double, regularyukawapot_single,
    regularyukawapot_double


@testset "CuNESSie.jl" begin
    @testset "device" begin
        @testset "math.jl" begin include("device/math.jl") end
        @testset "laplace.jl" begin include("device/laplace.jl") end
        @testset "yukawa.jl" begin include("device/yukawa.jl") end
    end
end
