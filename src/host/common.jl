@inline _kcfg(n::Int) = _ -> (threads = 128, blocks = cld(n, 128))

function _solve_linear_system(A::AbstractArray{T, 2}, b::AbstractArray{T, 1}) where T
    gmres(A, b,
        verbose=true,
        restart=min(200, size(A, 2)),
        Pl=DiagonalPreconditioner(A)
    )
end