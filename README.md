# CUDA-accelerated Nonlocal Electrostatics
[![](https://img.shields.io/github/license/tkemmer/CuNESSie.jl?style=for-the-badge)](https://github.com/tkemmer/CuNESSie.jl/blob/master/LICENSE)

`CuNESSie.jl` is an extension to the [`NESSie.jl`](https://github.com/tkemmer/NESSie.jl) package, providing CUDA-accelerated drop-in replacements for the package's numerical solvers and post-processors.


## Installation
In the Julia shell, switch to the
`Pkg` shell by pressing `]` and enter the following command:

```sh
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

## Citing
If you use `CuNESSie.jl` in your research, please cite the following publications:
> T. Kemmer, S. Hack, B. Schmidt, A. Hildebrandt. CUDA-accelerated protein electrostatics in linear
> space. Journal of Computational Science 70 (2023) 102022. <https://doi.org/10.1016/j.jocs.2023.102022>

> T. Kemmer. Space-efficient and exact system representations for the nonlocal protein electrostatics
> problem. Ph. D. thesis (2021), Johannes Gutenberg University Mainz. Mainz, Germany. <https://doi.org/10.25358/openscience-5689>

Citation items for BibTeX can be found in [CITATION.bib](https://github.com/tkemmer/CuNESSie.jl/blob/master/CITATION.bib).
