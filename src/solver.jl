function solve(
         ::Type{NonlocalES},
    model::Model{T, Triangle{T}}
) where T
    elements = CuArray(
        unpack([[e.v1; e.v2; e.v3; e.normal; e.distorig; e.area] for e in model.elements])
    )
    Ξ        = CuArray(unpack([e.center for e in model.elements]))
    numelem  = length(model.elements)

    M = SystemMatrix(Ξ, elements, numelem, model.params)
    idrs(M, ones(T, 3numelem), verbose=true)
end
