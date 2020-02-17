function solve(
         ::Type{NonlocalES},
    model::Model{T, Triangle{T}}
) where T
    elements = CuArray(
        unpack([[e.v1; e.v2; e.v3; e.normal; e.distorig; e.area] for e in model.elements])
    )
    Ξ        = CuArray(unpack([e.center for e in model.elements]))
    numelem  = length(model.elements)
    params   = model.params
    umol     = params.εΩ \   φmol(model)
    qmol     = params.εΩ \ ∂ₙφmol(model)

    rhs = zeros(T, 3numelem)
    rhs[1:numelem] .= righthandside(Ξ, elements, umol, qmol, numelem, params)

    M = SystemMatrix(Ξ, elements, numelem, params)

    idrs(M, rhs, verbose=true)
end
