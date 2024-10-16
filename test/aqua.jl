@testitem "Aqua" begin
    using Aqua

    Aqua.test_all(CuNESSie;
        persistent_tasks=false # requires NESSie to be registered
    )
end
