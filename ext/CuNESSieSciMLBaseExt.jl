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

# Avoid method ambiguity (since SciMLBase v3.15.0)
# https://github.com/SciML/SciMLBase.jl/pull/1365
@static if hasmethod(Base.:*, Tuple{AbstractMatrix, AbstractNoTimeSolution})

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
end

end # module
