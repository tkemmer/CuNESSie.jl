using CuNESSie
using CuArrays
using CUDAnative
using NESSie
using ImplicitArrays
using Test

using CuNESSie: Ξ2device, elements2device

@testset "CuNESSie.jl" begin
    @testset "base" begin
        @testset "math.jl" begin include("base/math.jl") end
        @testset "laplace.jl" begin include("base/laplace.jl") end
        @testset "yukawa.jl" begin include("base/yukawa.jl") end
    end

    @testset "nonlocal" begin
        @testset "matrix.jl" begin include("nonlocal/matrix.jl") end
        @testset "rhs.jl" begin include("nonlocal/rhs.jl") end
    end
end
