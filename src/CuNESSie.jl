module CuNESSie

using Adapt
using CuArrays
using CUDAnative
using IterativeSolvers
using LinearAlgebra
using NESSie
using NESSie: yukawa
using Preconditioners

# device code
include("device/model.jl")
include("device/math.jl")
include("device/matrix.jl")
include("device/laplace.jl")
include("device/yukawa.jl")
export CuPosition, CuTriangle, Ξ2device, elements2device,
    laplacepot_single, laplacepot_double,
    regularyukawapot_single, regularyukawapot_double

# common host code
include("host/model.jl")
include("host/matrix.jl")
include("host/common.jl")
include("host/local.jl")
include("host/post.jl")
export LaplacePotentialMatrix, LocalSystem, LocalSystemMatrix, rfenergy, solve

# nonlocal BEM
include("nonlocal/matrix.jl")
include("nonlocal/rhs.jl")
include("nonlocal/solver.jl")
export NonlocalSystem, NonlocalSystemMatrix, NonlocalSystemOutputs, solve

end # module
