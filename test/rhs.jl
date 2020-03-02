using CuNESSie: _rhs_k_kernel!, _rhs_v_kernel!
using NESSie: yukawa

@testset "_rhs_k_kernel!" begin
    # FIXME
    for T in [Float64]
        opt = defaultopt(T)
        yuk = yukawa(opt)

        v1 = T[1, -1, -1]
        v2 = T[0, 1, -1]
        v3 = T[-1, -1, -1]
        v4 = T[0, 0, 1]

        elements = [
            Triangle(v1, v2, v3),
            Triangle(v1, v2, v4),
            Triangle(v2, v3, v4),
            Triangle(v3, v1, v4)
        ]
        cuelements = _elem2cuarr(elements...)

        Ξ = [e.center for e in elements]
        cuΞ = _pos2cuxi(Ξ...)

        numelem = length(elements)
        dst = CuArray{T}(undef, numelem)

        @cuda threads=numelem _rhs_k_kernel!(
            dst, cuelements, cuΞ, CuArrays.ones(T, numelem), numelem,
            one(yuk) - (opt.εΩ / opt.εΣ), yuk
        )
        cures = Array(dst)

        Ks = InteractionMatrix(
            collect(enumerate(Ξ)),
            collect(enumerate(elements)),
            NESSie.BEM.KSfun{T}(opt)
        )
        @test cures ≈ Ks * ones(T, numelem)
    end
end

@testset "_rhs_v_kernel!" begin
    for T in [Float32, Float64]
        opt = defaultopt(T)
        yuk = yukawa(opt)

        v1 = T[1, -1, -1]
        v2 = T[0, 1, -1]
        v3 = T[-1, -1, -1]
        v4 = T[0, 0, 1]

        elements = [
            Triangle(v1, v2, v3),
            Triangle(v1, v2, v4),
            Triangle(v2, v3, v4),
            Triangle(v3, v1, v4)
        ]
        cuelements = _elem2cuarr(elements...)

        Ξ = [e.center for e in elements]
        cuΞ = _pos2cuxi(Ξ...)

        numelem = length(elements)
        dst = CuArray{T}(undef, numelem)

        @cuda threads=numelem _rhs_v_kernel!(
            dst, cuelements, cuΞ, CuArrays.ones(T, numelem), numelem,
            opt.εΩ * (one(yuk)/opt.εΣ - one(yuk)/opt.ε∞), opt.εΩ / opt.ε∞, yuk
        )
        cures = Array(dst)

        Vs = InteractionMatrix(
            Ξ,
            elements,
            NESSie.BEM.VSfun{T}(opt)
        )
        @test cures ≈ Vs * ones(T, numelem)
    end
end
