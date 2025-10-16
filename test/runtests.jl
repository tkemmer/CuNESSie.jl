using ParallelTestRunner
import CuNESSie
import CUDA

const init_code = quote
    using CuNESSie
end

testsuite = find_tests(pwd())
if !CUDA.functional()
    @warn "Failed to initialize CUDA. Skipping device tests..."
    testsuite = Dict(filter(p -> !startswith(p.first, "device/"), pairs(testsuite)))
end

runtests(CuNESSie, ["--verbose", ARGS...]; init_code, testsuite)
