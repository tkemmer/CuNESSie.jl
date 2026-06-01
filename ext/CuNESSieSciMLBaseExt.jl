module CuNESSieSciMLBaseExt

using NESSie:
    DoubleLayer,
    SingleLayer
using CuNESSie:
    LaplacePotentialMatrix,
    LocalSystemMatrix,
    NonlocalSystemMatrix,
    ReYukawaPotentialMatrix
using SciMLBase: AbstractNoTimeSolution

function Base.:*(
    A::LaplacePotentialMatrix{T, DoubleLayer},
    x::AbstractNoTimeSolution{T, 1}
) where T
    A * x.u
end

function Base.:*(
    A::LaplacePotentialMatrix{T, SingleLayer},
    x::AbstractNoTimeSolution{T, 1}
) where T
    A * x.u
end

function Base.:*(
    A::LocalSystemMatrix{T},
    x::AbstractNoTimeSolution{T, 1}
) where T
    A * x.u
end

function Base.:*(
    A::NonlocalSystemMatrix{T},
    x::AbstractNoTimeSolution{T, 1}
) where T
    A * x.u
end

function Base.:*(
    A::ReYukawaPotentialMatrix{T, DoubleLayer},
    x::AbstractNoTimeSolution{T, 1}
) where T
    A * x.u
end

function Base.:*(
    A::ReYukawaPotentialMatrix{T, SingleLayer},
    x::AbstractNoTimeSolution{T, 1}
) where T
    A * x.u
end

end # module
