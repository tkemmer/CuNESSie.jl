function solve(
         ::Type{NonlocalES},
    model::Model{T, Triangle{T}}
) where T
    Ξ = CuArray([
        [e.center[1] for e in model.elements];
        [e.center[2] for e in model.elements];
        [e.center[3] for e in model.elements]
    ])
    elements = CuArray(
        unpack([[e.v1; e.v2; e.v3; e.normal; e.distorig; e.area] for e in model.elements])
    )
    numelem  = length(model.elements)
    params   = model.params
    umol     = params.εΩ \   φmol(model)
    qmol     = params.εΩ \ ∂ₙφmol(model)

    rhs = zeros(T, 3numelem)
    rhs[1:numelem] .= righthandside(Ξ, elements, umol, qmol, numelem, params)

    M = NonlocalSystemMatrix(Ξ, elements, numelem, params)

    gmres(M, rhs, verbose=true, restart=min(200, size(M, 2)), Pl=DiagonalPreconditioner(M))
end