using CuNESSie
using CuArrays
using CUDAnative
using NESSie
using Test

@testset "CuNESSie.jl" begin
    @testset "laplace.jl" begin include("laplace.jl") end
    @testset "yukawa.jl" begin include("yukawa.jl") end
end
