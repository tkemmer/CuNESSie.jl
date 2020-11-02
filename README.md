# CUDA-accelerated Nonlocal Electrostatics
[![](https://img.shields.io/github/license/tkemmer/CuNESSie.jl?style=for-the-badge)](https://github.com/tkemmer/ImplicitArrays.jl/blob/master/LICENSE)

`CuNESSie.jl` is an extension to the [`NESSie.jl`](https://github.com/tkemmer/NESSie.jl) package, providing CUDA-accelerated drop-in replacements for the package's numerical solvers and post-processors.


## Installation
This package version requires Julia 1.0 or above. In the Julia shell, switch to the
`Pkg` shell by pressing `]` and enter the following command:

```sh
pkg> add https://github.com/tkemmer/NESSie.jl
pkg> add https://github.com/tkemmer/CuNESSie.jl
```


## Usage example
The basic usage of this package is the same as for `NESSie.jl`. Just replace the `NESSie.BEM` module by `CuNESSie` in your code and you're ready to go:

```julia
using NESSie
using CuNESSie  # before: NESSie.BEM
using NESSie.Format: readoff, readpqr

# I. Create model
model           = readoff("data/born/na.off")
model.charges   = readpqr("data/born/na.pqr")
model.params.εΩ = 1   # dielectric constant for vacuum model
model.params.εΣ = 78  # dielectric constant for water

# II. Apply nonlocal solver
bem = solve(NonlocalES, model)  # <-- CUDA-accelerated solver

# III. Apply postprocessor
val = rfenergy(bem)             # <-- CUDA-accelerated post-processor
println("Reaction field energy: $val kJ/mol")
```

`CuNESSie.jl` reuses the system models and solver results from `NESSie.jl`, so the local and nonlocal BEM solvers as well as the corresponding post-processors from both packages can be interchanged freely.

## Testing
`CuNESSie.jl` provides tests for most of its functions. You can run the test suite with the
following command in the `Pkg` shell:
```sh
pkg> test CuNESSie
```
