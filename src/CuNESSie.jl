module CuNESSie

using Adapt
using CuArrays
using CUDAnative
using IterativeSolvers
using LinearAlgebra
using NESSie
using NESSie: yukawa
using NESSie.BEM: BEMResult, LocalBEMResult, NonlocalBEMResult
using Preconditioners

# device code
include("device/model.jl")
include("device/math.jl")
include("device/matrix.jl")
include("device/laplace.jl")
include("device/yukawa.jl")
include("device/nonlocal.jl")
export CuPosition, CuPositionVector, CuTriangle, CuTriangleVector, NoAccessArray, ReadOnlyArray,
    laplacepot_single, laplacepot_double, regularyukawapot_single, regularyukawapot_double

# host code
include("host/common.jl")
include("host/model.jl")
include("host/matrix.jl")
include("host/local.jl")
include("host/nonlocal.jl")
include("host/post.jl")
export LaplacePotentialMatrix, LocalSystem, LocalSystemMatrix, NonlocalSystem,
    NonlocalSystemMatrix, NonlocalSystemOutputs, PositionVector, PostProcessor,
    ReYukawaPotentialMatrix, TriangleVector, rfenergy, solve, φ⃰Ω, φΩ, φΣ

end # module
