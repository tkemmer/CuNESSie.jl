using TestItemRunner
using CUDA

if CUDA.functional()
    @run_package_tests verbose=true
else
    @warn "Failed to initialize CUDA. Skipping device tests..."
    @run_package_tests verbose=true filter=ti->!(:device in ti.tags )
end
