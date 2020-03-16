using CuNESSie
using CuArrays
using CUDAnative
using NESSie
using ImplicitArrays
using Test

using CuNESSie: Ξ2device, elements2device

@testset "CuNESSie.jl" begin
    @testset "math.jl" begin include("math.jl") end
    @testset "laplace.jl" begin include("laplace.jl") end
    @testset "yukawa.jl" begin include("yukawa.jl") end
    @testset "matrix.jl" begin include("matrix.jl") end
    @testset "rhs.jl" begin include("rhs.jl") end
end
