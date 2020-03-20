const _etol_f64 = 1.45e-8
const _etol_f32 = 3.45f-4

@inline _etol(::Float64) = _etol_f64
@inline _etol(::Type{Float64}) = _etol_f64
@inline _etol(::Float32) = _etol_f32
@inline _etol(::Type{Float32}) = _etol_f32
