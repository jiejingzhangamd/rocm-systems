#include <hip/hip_runtime.h>

__global__ void register_lifetime_probe(float* x) { *x += 1.0f; }

extern "C" void launch_register_lifetime_probe(float* x)
{
    hipLaunchKernelGGL(register_lifetime_probe, dim3(1), dim3(1), 0, nullptr, x);
}
