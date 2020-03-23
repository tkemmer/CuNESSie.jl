struct LocalSystem{T}
    model   ::Model{T, Triangle{T}}
    Ξ       ::PositionVector{T}
    elements::TriangleVector{T}
    umol    ::Vector{T}
    qmol    ::Vector{T}
end

function LocalSystem(model::Model{T, Triangle{T}}) where T
    cuΞ    = PositionVector([e.center for e in model.elements])
    cuelms = TriangleVector(model.elements)
    umol   = model.params.εΩ .\ φmol(model, tolerance=_etol(T))
    qmol   = model.params.εΩ .\ ∂ₙφmol(model)
    LocalSystem(model, cuΞ, cuelms, umol, qmol)
end

struct LocalSystemMatrix{T} <: PotentialMatrix{T}
    K::LaplacePotentialMatrix{T, DoubleLayer}
    α::T
    β::T
end

@inline LocalSystemMatrix(
    K     ::LaplacePotentialMatrix{T, DoubleLayer},
    params::Option{T}
) where T = LocalSystemMatrix(K, (1 + params.εΩ / params.εΣ) * T(2π), params.εΩ / params.εΣ - 1)

@inline Base.size(A::LocalSystemMatrix{T}) where T = Base.size(A.K)

@inline Base.:*(
    A::LocalSystemMatrix{T},
    x::AbstractArray{T, 1}
) where T = A.α .* x .+ (A.β .* (A.K * x))

@inline function LinearAlgebra.diag(
    A::LocalSystemMatrix{T},
    k::Int = 0
) where T
    k != 0 && error("diag not defined for k != 0 on ", typeof(A))
    A.α .* ones(T, size(A, 1))
end

function solve(sys::LocalSystem{T}) where T
    εΩ = sys.model.params.εΩ
    εΣ = sys.model.params.εΣ
    K  = LaplacePotentialMatrix{DoubleLayer}(sys.Ξ, sys.elements)
    V  = LaplacePotentialMatrix{SingleLayer}(sys.Ξ, sys.elements)
    A  = LocalSystemMatrix(K, sys.model.params)

    u = _solve_linear_system(A, K * sys.umol .- (T(2π) .* sys.umol) .- (εΩ / εΣ .* (V * sys.qmol)))
    q = _solve_linear_system(V, K * u .+ (T(2π) .* u))

    LocalBEMResult(sys.model, u, q, sys.umol, sys.qmol)
end

@inline solve(
         ::Type{LocalES},
    model::Model{T, Triangle{T}}
) where T = solve(LocalSystem(model))
