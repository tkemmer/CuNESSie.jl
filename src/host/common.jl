const _etol_f64 = 1.45e-8
const _etol_f32 = 3.45f-4

@inline _etol(::Float64) = _etol_f64
@inline _etol(::Type{Float64}) = _etol_f64
@inline _etol(::Float32) = _etol_f32
@inline _etol(::Type{Float32}) = _etol_f32

@inline _kcfg(n::Int) = _ -> (threads = 128, blocks = cld(n, 128))
@inline _kcfg(A::PotentialMatrix{T}) where T = _kcfg(size(A, 1))

abstract type ReadOnlyArray{T, N} <: AbstractArray{T, N} end
abstract type NoAccessArray{T, N} <: AbstractArray{T, N} end

@inline Base.getindex(A::NoAccessArray, ::Int) = error("getindex not defined for ", typeof(A))

@inline Base.setindex!(
    A::Union{ReadOnlyArray, NoAccessArray},
     ::Any,
     ::Int
) = error("setindex! not defined for ", typeof(A))

@inline Base.show(io::IO, ::MIME"text/plain", A ::NoAccessArray) = Base.show(io, A)
@inline Base.show(io::IO, A::NoAccessArray) = print(io, join(size(A), "×"), " ", typeof(A))

function _solve_linear_system(A::AbstractArray{T, 2}, b::AbstractArray{T, 1}) where T
    gmres(A, b,
        verbose=true,
        restart=min(200, size(A, 2)),
        Pl=DiagonalPreconditioner(A)
    )
end
