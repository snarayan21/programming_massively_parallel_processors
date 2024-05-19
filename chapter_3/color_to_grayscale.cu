#include <iostream>

__global__
// The image is encoded as unsigned chars -- unsigned ints from 0 to 255.
void colorToGrayscaleKernel(unsigned char* Pout, unsigned char* Pin, int width, int height){
    // successive rows are in the y direction, successive columns are in the x direction.
    // same as going over the matrix horizontally = increase x
    // same as going over the matrix vertically = increase y
    int col_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int row_idx = blockDim.y * blockIdx.y + threadIdx.y;

    // Only convert the pixel if it's within the image bounds
    if (col_idx < width && row_idx < height){
        int grayscaleOffset = row_idx * width + col_idx;

        // In the color image, the pixels are stored as 3-tuples of unsigned chars (I think...)
        // So the pixel offset is 3 times the grayscale offset, and the order is R, G, B
        int colorOffset = grayscaleOffset * 3;
        unsigned char r = Pin[colorOffset];
        unsigned char g = Pin[colorOffset + 1];
        unsigned char b = Pin[colorOffset + 2];

        // The formula for converting color to grayscale is 0.21*r + 0.72*g + 0.07*b
        Pout[grayscaleOffset] = 0.21f * r + 0.72f * g + 0.07f * b;
    }
}

void colorToGrayscale(unsigned char* Pout_h, unsigned char* Pin_h, int width, int height){

    // Initialize and allocate pointers to device memory.
    unsigned char *Pout_d, *Pin_d;
    int grayscale_size = width * height * sizeof(unsigned char);

    // Allocate memory for the input (color) and output (grayscale) images
    // There are 3 channels, R, G, and B.
    cudaMalloc((void**) &Pin_d, 3 * grayscale_size);
    cudaMalloc((void**) &Pout_d, grayscale_size);

    // Copy input image from host memory to device memory
    cudaMemcpy(Pin_d, Pin_h, 3 * grayscale_size, cudaMemcpyHostToDevice);
    // Copy output image from host memory to device memory
    cudaMemcpy(Pout_d, Pout_h, grayscale_size, cudaMemcpyHostToDevice);

    // Launch the kernel
    // Use a 2D grid of 2D blocks, each 16x16.
    // when defining these sizes in the stub host-side, the order is x, y, z.
    // But indexing is in reverse order -- z is outermost, then y, then x.
    // z "selecting" the plane of the matrix on which to operate, and
    // y and x selecting the vertical and horizontal offsets, respectively,
    // helps me visualize the indexing a bit better.
    dim3 dimGrid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 dimBlock(16, 16, 1);

    colorToGrayscaleKernel<<<dimGrid, dimBlock>>>(Pout_d, Pin_d, width, height);

    // Copy result from device memory to host memory
    cudaMemcpy(Pout_h, Pout_d, grayscale_size, cudaMemcpyDeviceToHost);

    // Don't forget to free device memory!!
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

int main(){
    std::cout << "Color to Grayscale" << std::endl;

    int width = 72;
    int height = 60;
    int total_px = width * height;

    unsigned char* Pout_h = new unsigned char[total_px];
    unsigned char* Pin_h = new unsigned char[3*total_px];

    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            std::cout << "i: " << i << " j: " << j << std::endl;
            Pout_h[i*width + j] = (unsigned char) (i+j) % 256;
        }
    }

    colorToGrayscale(Pout_h, Pin_h, width, height);

    std::cout << "Color Image:";
    for (int i = 0; i < height; i++){
        for (int j = 0; j < 3*width; j++){
            std::cout << (int) Pin_h[i*3*width + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Grayscale Image:";
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            std::cout << (int) Pout_h[i*width + j] << " ";
        }
        std::cout << std::endl;
    }
}