module CuNESSie

using Adapt
using CUDA
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

# host code
include("host/common.jl")
include("host/model.jl")
include("host/matrix.jl")
include("host/local.jl")
include("host/nonlocal.jl")
include("host/post.jl")
export rfenergy, solve, φΩ, φΣ

end # module
