module CuNESSie

using CuArrays
using CUDAnative

include("math.jl")
include("laplace.jl")

greet() = print("Hello World!")

end # module
