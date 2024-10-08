#include <iostream>
#include <algorithm>
#include <cstdio>

// kernel function
__global__
void matVecKernel(float* Vout, float* Vin, float* Mat, int len){
    // get col and row indices of the matvec. Each thread computes one element of the output.
    // this is just a 1D grid....
    // we are also assuming that Mat is a square matrix, meaning that Vout and Vin are the same length.
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Bounds checking for the length of the vector.
    if (idx < len){
        // For output element at idx, we want to dot product the idx-th row of Mat by Vin
        float accum = 0.0f;
        for (int i = 0; i < len; i++){
            accum += Mat[idx*len + i] * Vin[i];
        }
        Vout[idx] = accum;
    }
}

// stub function
void matVec(float* Vout_h, float* Vin_h, float* Mat_h, int len){

    // Initialize and allocate pointers to device memory.
    float *Vout_d, *Vin_d, *Mat_d;

    // Allocate memory for Vin_d, Vout_d, Mat_d on device
    cudaMalloc((void**) &Vin_d, len*sizeof(float));
    cudaMalloc((void**) &Vout_d, len*sizeof(float));
    cudaMalloc((void**) &Mat_d, len*len*sizeof(float));

    // Copies from host memory to device memory
    cudaMemcpy(Vin_d, Vin_h, len*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Mat_d, Mat_h, len*len*sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    // Use a 1D grid of blocks, each with 256 threads.
    matVecKernel<<<ceil(len/256.0), 256>>>(Vout_d, Vin_d, Mat_d, len);

    // Copy result from device memory to host memory
    cudaMemcpy(Vout_h, Vout_d, len*sizeof(float), cudaMemcpyDeviceToHost);

    // Don't forget to free device memory!!
    cudaFree(Vin_d);
    cudaFree(Vout_d);
    cudaFree(Mat_d);
}

int main(){
    std::cout << "MatVec" << std::endl;

    int len = 125;

    float* Vout_h = new float[len];
    float* Vin_h = new float[len];
    float* Mat_h = new float[len*len];
    
    // Mat_h is all 1s
    std::fill(Mat_h, Mat_h + len*len, 1.0f);

    // Vin_h will be increasing values
    for (int i = 0; i < len; i++){
        Vin_h[i] = (float) i;
    }

    std::cout << "Input Vector: ";
    for (int i = 0; i < len; i++) {
        std::cout << (float) Vin_h[i] << " ";        
    }
    std::cout << std::endl;

    // call stub which takes care of everything device-side
    matVec(Vout_h, Vin_h, Mat_h, len);


    std::cout << "Output Vector: ";
    for (int i = 0; i < len; i++) {
        std::cout << (float) Vout_h[i] << " ";        
    }
    std::cout << std::endl;
}