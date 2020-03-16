module CuNESSie

using CuArrays
using CUDAnative
using IterativeSolvers
using LinearAlgebra
using NESSie
using NESSie: unpack, yukawa
using Preconditioners

include("common.jl")
include("model.jl")
include("math.jl")
include("laplace.jl")
include("yukawa.jl")

# nonlocal BEM
include("nonlocal/matrix.jl")
include("nonlocal/rhs.jl")
include("nonlocal/solver.jl")

end # module
