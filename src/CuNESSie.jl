module CuNESSie

using CuArrays
using CUDAnative
using IterativeSolvers
using LinearAlgebra
using NESSie
using NESSie: yukawa
using Preconditioners

# device code
include("device/common.jl")
include("device/model.jl")
include("device/math.jl")
include("device/matrix.jl")
include("device/laplace.jl")
include("device/yukawa.jl")
export CuPosition, CuTriangle, Ξ2device, elements2device,
    laplacepot_single, laplacepot_double,
    regularyukawapot_single, regularyukawapot_double

# common host code
include("host/common.jl")
include("host/matrix.jl")
export LaplacePotentialMatrix

# nonlocal BEM
include("nonlocal/matrix.jl")
include("nonlocal/rhs.jl")
include("nonlocal/solver.jl")
include("nonlocal/post.jl")
export NonlocalSystem, NonlocalSystemMatrix, NonlocalSystemOutputs, rfenergy, solve

end # module
