using CuNESSie: matvec11!, matvec12!, matvec13!, matvec21!, matvec22!, matvec32!, matvec33!
using NESSie: yukawa

@testset "matvec11!" begin
    # FIXME
    @test_skip "Float32 is actually more precise here than with NESSie"
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
        cuΞ = CuArray(NESSie.unpack(Ξ))

        numelem = length(elements)
        dst = CuArray{T}(undef, numelem)

        @cuda threads=numelem matvec11!(
            dst, cuelements, cuΞ, CuArrays.ones(T, numelem), numelem, yuk
        )
        cures = Array(dst)

        M₁₁ = InteractionMatrix(
            collect(enumerate(Ξ)),
            collect(enumerate(elements)),
            NESSie.BEM.M11fun{T}(opt)
        )
        @test cures ≈ M₁₁ * ones(T, numelem)
    end
end

@testset "matvec12!" begin
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
        cuΞ = CuArray(NESSie.unpack(Ξ))

        numelem = length(elements)
        dst = CuArray{T}(undef, numelem)

        @cuda threads=numelem matvec12!(
            dst, cuelements, cuΞ, CuArrays.ones(T, numelem), numelem,
            opt.εΩ * (1/opt.ε∞ - 1/opt.εΣ), opt.εΩ/opt.ε∞, yuk
        )
        cures = Array(dst)

        M₁₂ = InteractionMatrix(Ξ, elements, NESSie.BEM.M12fun{T}(opt))

        @test cures ≈ M₁₂ * ones(T, numelem)
    end
end

@testset "matvec13!" begin
    # FIXME
    @test_skip "Float32 is actually more precise here than with NESSie"
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
        cuΞ = CuArray(NESSie.unpack(Ξ))

        numelem = length(elements)
        dst = CuArray{T}(undef, numelem)

        @cuda threads=numelem matvec13!(
            dst, cuelements, cuΞ, CuArrays.ones(T, numelem), numelem, opt.ε∞/opt.εΣ, yuk
        )
        cures = Array(dst)

        M₁₃ = InteractionMatrix(Ξ, elements, NESSie.BEM.M13fun{T}(opt))

        @test cures ≈ M₁₃ * ones(T, numelem)
    end
end

@testset "matvec21!" begin
    for T in [Float32, Float64]
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
        cuΞ = CuArray(NESSie.unpack(Ξ))

        numelem = length(elements)
        dst = CuArray{T}(undef, numelem)

        @cuda threads=numelem matvec21!(
            dst, cuelements, cuΞ, CuArrays.ones(T, numelem), numelem
        )
        cures = Array(dst)

        M₂₁ = InteractionMatrix(
            collect(enumerate(Ξ)),
            collect(enumerate(elements)),
            NESSie.BEM.M21fun{T}()
        )

        @test cures ≈ M₂₁ * ones(T, numelem)
    end
end

@testset "matvec22!" begin
    for T in [Float32, Float64]
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
        cuΞ = CuArray(NESSie.unpack(Ξ))

        numelem = length(elements)
        dst = CuArray{T}(undef, numelem)

        @cuda threads=numelem matvec22!(
            dst, cuelements, cuΞ, CuArrays.ones(T, numelem), numelem
        )
        cures = Array(dst)

        M₂₂ = InteractionMatrix(Ξ, elements, NESSie.BEM.M22fun{T}())

        @test cures ≈ M₂₂ * ones(T, numelem)
    end
end

@testset "matvec32!" begin
    for T in [Float32, Float64]
        opt = defaultopt(T)

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
        cuΞ = CuArray(NESSie.unpack(Ξ))

        numelem = length(elements)
        dst = CuArray{T}(undef, numelem)

        @cuda threads=numelem matvec32!(
            dst, cuelements, cuΞ, CuArrays.ones(T, numelem), numelem, opt.εΩ/opt.ε∞
        )
        cures = Array(dst)

        M₃₂ = InteractionMatrix(Ξ, elements, NESSie.BEM.M32fun{T}(opt))

        @test cures ≈ M₃₂ * ones(T, numelem)
    end
end

@testset "matvec33!" begin
    for T in [Float32, Float64]
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
        cuΞ = CuArray(NESSie.unpack(Ξ))

        numelem = length(elements)
        dst = CuArray{T}(undef, numelem)

        @cuda threads=numelem matvec33!(
            dst, cuelements, cuΞ, CuArrays.ones(T, numelem), numelem
        )
        cures = Array(dst)

        M₃₃ = InteractionMatrix(
            collect(enumerate(Ξ)),
            collect(enumerate(elements)),
            NESSie.BEM.M33fun{T}()
        )

        @test cures ≈ M₃₃ * ones(T, numelem)
    end
end
