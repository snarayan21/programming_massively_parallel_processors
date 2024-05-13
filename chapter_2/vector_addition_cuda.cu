#include <iostream>

// __global__ keyword tells compiler that the function is a kernel, and invoking it will launch
// a grid of threads. Sometimes kernels can be called on device, but mostly they will be called from
// host to execute on device.
__global__
void vecAddKernel(float* A, float* B, float* C, int n){
    // This gives us the index of the element i the arrays that should get added together.
    // blockDim, blockIdx, and threadIdx are built-in variables that are available in kernels
    // and give us the ability to identify which thread & thread block we are in, and the index
    // of the current thread within the block.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Now, we simply add together the element this thread is responsible for.
    // We make sure to check if index i is within bounds -- if it's less than n.
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n)
{
    // Initialize and allocate pointers to device memory
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);
    // First arg is a pointer to the pointer for each object
    // Why? cudaMalloc will set a new value for the pointer -- an address in device memory, 
    // instead of the address in host memory
    // Second arg is the size to allocate, in bytes.
    cudaMalloc((void**) &A_d, size);
    cudaMalloc((void**) &B_d, size);
    cudaMalloc((void**) &C_d, size);

    // Copy input vectors from host memory to device memory
    // First arg is the destination, second is the source, third is the size, fourth is the direction
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
    // We don't copy C_h to C_d because C_d will be the output of the kernel. There's nothing of use to copy.

    // Kernel invocation goes here!!!
    // Launching kernel with 256 threads in each block, meaning n/256 blocks total.
    // first number is for # blocks, second is # threads in each block.
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    // Copy result from device memory C_d to host memory C_h
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free device memory
    // cudaFree only needs the pointers to device memory, not a pointer to the pointers to device memory,
    // since it just needs to free the memory at the address, not change the address itself. (unlike cudaMalloc)
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

}

int main()
{
    std::cout << "Naive Vector Addition" << std::endl;

    int n = 1 << 5;

    float* A_h = new float[n];
    float* B_h = new float[n];
    float* C_h = new float[n];

    for (int i = 0; i < n; i++)
    {
        A_h[i] = i*1.0f;
        B_h[i] = i*2.0f;
    }

    vecAdd(A_h, B_h, C_h, n);

    std::cout << "A: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << A_h[i] << " ";
    }

    std::cout << std::endl << "B: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << B_h[i] << " ";
    }

    std::cout << std::endl << "C: ";
    for (int i = 0; i < n; i++)
    {
        std::cout << C_h[i] << " ";
    }

    std::cout << std::endl;

}