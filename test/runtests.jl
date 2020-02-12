using CuNESSie
using CuArrays
using CUDAnative
using NESSie
using Test

@inline function _elem2cuarr(elements::Triangle{T}...) where T
    CuArray(
        NESSie.unpack([[e.v1; e.v2; e.v3; e.normal; e.distorig; e.area] for e in elements])
    )
end

@testset "CuNESSie.jl" begin
    @testset "laplace.jl" begin include("laplace.jl") end
    @testset "yukawa.jl" begin include("yukawa.jl") end
end
