#include <iostream>

int main()
{
    std::cout << "CUDA Device Properties" << std::endl;
    std::cout << std::endl;

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    cudaDeviceProp deviceProp;
    for (int i = 0; i < deviceCount; i++)
    {
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "Clock rate: " << deviceProp.clockRate << std::endl;
        std::cout << "Max threads by dimension (x, y, z): " << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << std::endl;
        std::cout << "Max registers per block (should be same as for SM): " << deviceProp.regsPerBlock << std::endl;
        std::cout << "Number of threads per warp: " << deviceProp.warpSize << std::endl;
        std::cout << std::endl;
    }

    return 0;
}