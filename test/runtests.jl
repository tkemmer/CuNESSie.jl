using CuNESSie
using CuArrays
using CUDAnative
using NESSie
using ImplicitArrays
using Test

@testset "CuNESSie.jl" begin
    @testset "device" begin
        @testset "math.jl" begin include("device/math.jl") end
        @testset "laplace.jl" begin include("device/laplace.jl") end
        @testset "yukawa.jl" begin include("device/yukawa.jl") end
    end

    @testset "nonlocal" begin
        @testset "matrix.jl" begin include("nonlocal/matrix.jl") end
        @testset "rhs.jl" begin include("nonlocal/rhs.jl") end
    end
end
