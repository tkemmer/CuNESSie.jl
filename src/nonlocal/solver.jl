struct NonlocalSystem{T}
    model::Model{T, Triangle{T}}
    A    ::NonlocalSystemMatrix{T}
    b    ::NonlocalSystemOutputs{T}
    umol ::Vector{T}
    qmol ::Vector{T}
end

function NonlocalSystem(model::Model{T, Triangle{T}}) where T
    cuΞ    = PositionVector([e.center for e in model.elements])
    cuelms = TriangleVector(model.elements)

    A    = NonlocalSystemMatrix(cuΞ, cuelms, model.params)
    umol = model.params.εΩ .\   φmol(model, tolerance=_etol(T))
    qmol = model.params.εΩ .\ ∂ₙφmol(model)
    b    = NonlocalSystemOutputs(A, umol, qmol)

    NonlocalSystem(model, A, b, umol, qmol)
end

function solve(sys::NonlocalSystem{T}) where T
    numξ = length(sys.A.Ξ)
    cauchy  = _solve_linear_system(sys.A, sys.b)
    NonlocalBEMResult(
        sys.model,
        view(cauchy, 1:numξ),
        view(cauchy, 1+numξ:2numξ),
        view(cauchy, 1+numξ:3numξ),
        sys.umol,
        sys.qmol
    )
end

@inline solve(
         ::Type{NonlocalES},
    model::Model{T, Triangle{T}}
) where T = solve(NonlocalSystem(model))
