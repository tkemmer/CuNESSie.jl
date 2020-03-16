module CuNESSie

using CuArrays
using CUDAnative
using IterativeSolvers
using LinearAlgebra
using NESSie
using NESSie: yukawa
using Preconditioners

include("base/common.jl")
include("base/model.jl")
include("base/math.jl")
include("base/laplace.jl")
include("base/yukawa.jl")
export CuPosition, CuTriangle, Ξ2device, elements2device,
    laplacepot_single, laplacepot_double,
    regularyukawapot_single, regularyukawapot_double

# nonlocal BEM
include("nonlocal/matrix.jl")
include("nonlocal/rhs.jl")
include("nonlocal/solver.jl")
export NonlocalSystem, NonlocalSystemMatrix, NonlocalSystemOutputs, solve

end # module
