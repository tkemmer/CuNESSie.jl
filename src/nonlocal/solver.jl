struct NonlocalSystem{T}
    A   ::NonlocalSystemMatrix{T}
    b   ::NonlocalSystemOutputs{T}
    umol::Vector{T}
    qmol::Vector{T}
end

function NonlocalSystem(model::Model{T, Triangle{T}}) where T
    Ξ = CuArray([
        [e.center[1] for e in model.elements];
        [e.center[2] for e in model.elements];
        [e.center[3] for e in model.elements]
    ])
    elements = CuArray(
        unpack([[e.v1; e.v2; e.v3; e.normal; e.distorig; e.area] for e in model.elements])
    )

    A    = NonlocalSystemMatrix(Ξ, elements, length(model.elements), model.params)
    umol = model.params.εΩ .\   φmol(model)
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
