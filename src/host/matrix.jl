abstract type PotentialMatrix{T} <: AbstractArray{T, 2} end

@inline Base.getindex(
    A::PotentialMatrix{T},
     ::Int
) where T = error("getindex not defined for", typeof(A))

@inline Base.setindex!(
    A::PotentialMatrix{T},
     ::Any,
     ::Int
) where T = error("setindex! not defined for", typeof(A))

@inline Base.show(
    io::IO,
      ::MIME"text/plain",
    A ::PotentialMatrix{T}
) where T = Base.show(io, A)


@inline function Base.show(io::IO, A::PotentialMatrix{T}) where {T}
    d = join(size(A), "×")
    print(io, "$d ", typeof(A))
end

@inline LinearAlgebra.mul!(
    Y::AbstractArray{T, 1},
    A::PotentialMatrix{T},
    x::AbstractArray{T, 1}
) where T = Y .= A * x

struct LaplacePotentialMatrix{T, L <: PotentialType} <: PotentialMatrix{T}
    Ξ       ::PositionVector{T}
    elements::TriangleVector{T}
end

@inline LaplacePotentialMatrix{L}(
    Ξ       ::PositionVector{T},
    elements::TriangleVector{T}
) where {T, L <: PotentialType} = LaplacePotentialMatrix{T, L}(Ξ, elements)

@inline Base.size(A::LaplacePotentialMatrix{T}) where T = (length(A.Ξ), length(A.elements))

@inline Base.:*(
    A::LaplacePotentialMatrix{T, SingleLayer},
    x::AbstractArray{T, 1}
) where T = _mul(A, x, _laplace_single_mul_kernel!)

@inline Base.:*(
    A::LaplacePotentialMatrix{T, DoubleLayer},
    x::AbstractArray{T, 1}
) where T = _mul(A, x, _laplace_double_mul_kernel!)

@inline LinearAlgebra.diag(
    A::LaplacePotentialMatrix{T, SingleLayer},
    k::Int = 0
) where T = _diag(A, k, _laplace_single_diag_kernel!)

@inline LinearAlgebra.diag(
    A::LaplacePotentialMatrix{T, DoubleLayer},
    k::Int = 0
) where T = _diag(A, k, _laplace_double_diag_kernel!)

function _diag(A::LaplacePotentialMatrix{T}, k::Int, pot::F) where {T, F <: Function}
    @assert size(A, 1) == size(A, 2) "diag requires a square matrix"
    k != 0 && error("diag not defined for k != 0 on ", typeof(A))
    dst = CuArray{T}(undef, size(A, 1))
    @cuda config=_kcfg(A) pot(dst, A.Ξ, A.elements)
    Array(dst)
end

function _mul(A::LaplacePotentialMatrix{T}, x::AbstractArray{T, 1}, pot::F) where {T, F <: Function}
    dst = CuArray{T}(undef, size(A, 1))
    @cuda config=_kcfg(A) pot(dst, A.Ξ, A.elements, CuArray(x))
    Array(dst)
end

struct ReYukawaPotentialMatrix{T, L <: PotentialType} <: PotentialMatrix{T}
    Ξ       ::PositionVector{T}
    elements::TriangleVector{T}
    yuk     ::T
end

@inline ReYukawaPotentialMatrix{L}(
    Ξ       ::PositionVector{T},
    elements::TriangleVector{T},
    yuk     ::T
) where {T, L <: PotentialType} = ReYukawaPotentialMatrix{T, L}(Ξ, elements, yuk)

@inline Base.size(A::ReYukawaPotentialMatrix{T}) where T = (length(A.Ξ), length(A.elements))

@inline Base.:*(
    A::ReYukawaPotentialMatrix{T, SingleLayer},
    x::AbstractArray{T, 1}
) where T = _mul(A, x, _reyukawa_single_mul_kernel!)

@inline Base.:*(
    A::ReYukawaPotentialMatrix{T, DoubleLayer},
    x::AbstractArray{T, 1}
) where T = _mul(A, x, _reyukawa_double_mul_kernel!)

function _mul(
    A  ::ReYukawaPotentialMatrix{T},
    x  ::AbstractArray{T, 1},
    pot::F
) where {T, F <: Function}
    dst = CuArray{T}(undef, size(A, 1))
    @cuda config=_kcfg(A) pot(dst, A.Ξ, A.elements, CuArray(x), A.yuk)
    Array(dst)
end
