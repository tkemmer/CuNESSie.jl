struct NonlocalSystem{T}
    model::Model{T, Triangle{T}}
    A    ::NonlocalSystemMatrix{T}
    b    ::NonlocalSystemOutputs{T}
    umol ::Vector{T}
    qmol ::Vector{T}
end

function NonlocalSystem(model::Model{T, Triangle{T}}) where T
    cuΞ    = Ξ2device([e.center for e in model.elements])
    cuelms = elements2device(model.elements)

    A    = NonlocalSystemMatrix(cuΞ, cuelms, length(model.elements), model.params)
    umol = model.params.εΩ .\   φmol(model, tolerance=_etol(T))
    qmol = model.params.εΩ .\ ∂ₙφmol(model)
    b    = NonlocalSystemOutputs(A, umol, qmol)

    NonlocalSystem(model, A, b, umol, qmol)
end

function solve(sys::NonlocalSystem{T}) where T
    numelem = sys.A.numelem
    cauchy  = _solve_linear_system(sys.A, sys.b)
    NESSie.BEM.NonlocalBEMResult(
        sys.model,
        view(cauchy, 1:numelem),
        view(cauchy, 1+numelem:2numelem),
        view(cauchy, 1+2numelem:3numelem),
        sys.umol,
        sys.qmol
    )
end

@inline solve(
         ::Type{NonlocalES},
    model::Model{T, Triangle{T}}
) where T = solve(NonlocalSystem(model))
