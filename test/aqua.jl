@testitem "Aqua" begin
    using Aqua

    # Current workaround to avoid failing tests due to ambiguity in dependencies
    # https://github.com/JuliaTesting/Aqua.jl/issues/77
    @testset "Method ambiguity" begin
        Aqua.test_ambiguities(CuNESSie)
    end

    Aqua.test_all(CuNESSie;
        ambiguities=false,
        persistent_tasks=false # requires NESSie to be registered
    )
end
