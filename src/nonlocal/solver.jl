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

function _solve(sys::NonlocalSystem{T}) where T
    gmres(sys.A, sys.b,
        verbose=true,
        restart=min(200, size(sys.A, 2)),
        Pl=DiagonalPreconditioner(sys.A)
    )
end

function solve(sys::NonlocalSystem{T}) where T
    numelem = sys.A.numelem
    cauchy  = _solve(sys)
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
