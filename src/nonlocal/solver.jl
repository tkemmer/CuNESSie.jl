struct NonlocalSystem{T}
    A   ::NonlocalSystemMatrix{T}
    b   ::NonlocalSystemOutputs{T}
    umol::Vector{T}
    qmol::Vector{T}
end

function NonlocalSystem(model::Model{T, Triangle{T}}) where T
    cuΞ    = Ξ2device([e.center for e in model.elements])
    cuelms = elements2device(model.elements)

    A    = NonlocalSystemMatrix(cuΞ, cuelms, length(model.elements), model.params)
    umol = model.params.εΩ .\   φmol(model, tolerance=_etol(T))
    qmol = model.params.εΩ .\ ∂ₙφmol(model)
    b    = NonlocalSystemOutputs(A, umol, qmol)

    NonlocalSystem(A, b, umol, qmol)
end

function solve(sys::NonlocalSystem{T}) where T
    gmres(sys.A, sys.b,
        verbose=true,
        restart=min(200, size(sys.A, 2)),
        Pl=DiagonalPreconditioner(sys.A)
    )
end

@inline solve(
         ::Type{NonlocalES},
    model::Model{T, Triangle{T}}
) where T = solve(NonlocalSystem(model))
