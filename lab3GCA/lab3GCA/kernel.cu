
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cstdint>      // Data types
#include <iostream>     // File operations

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 941       // VR width
#define N 704       // VR height
#define C 3         // Colors
#define OFFSET 15   // Header length

uint8_t* get_image_array(void) {
    /*
     * Get the data of an (RGB) image as a 1D array.
     *
     * Returns: Flattened image array.
     *
     * Noets:
     *  - Images data is flattened per color, column, row.
     *  - The first 3 data elements are the RGB components
     *  - The first 3*M data elements represent the firts row of the image
     *  - For example, r_{0,0}, g_{0,0}, b_{0,0}, ..., b_{0,M}, r_{1,0}, ..., b_{b,M}, ..., b_{N,M}
     *
     */
     // Try opening the file
    FILE* imageFile;
    imageFile = fopen("./input_image.ppm", "rb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Initialize empty image array
    uint8_t* image_array = (uint8_t*)malloc(M * N * C * sizeof(uint8_t) + OFFSET);

    // Read the image
    fread(image_array, sizeof(uint8_t), M * N * C * sizeof(uint8_t) + OFFSET, imageFile);

    // Close the file
    fclose(imageFile);

    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}

void save_image_array(uint8_t* image_array) {
    /*
     * Save the data of an (RGB) image as a pixel map.
     *
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     *
     */
     // Try opening the file
    FILE* imageFile;
    imageFile = fopen("./output_image.ppm", "wb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Configure the file
    fprintf(imageFile, "P6\n");               // P6 filetype
    fprintf(imageFile, "%d %d\n", M, N);      // dimensions
    fprintf(imageFile, "255\n");              // Max pixel

    // Write the image
    fwrite(image_array, 1, M * N * C, imageFile);

    // Close the file
    fclose(imageFile);
}

//flip kernel GPU
__global__ void flipImage(uint8_t* input, uint8_t* output, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        output[i] = 255 - input[i];
    }
}

void invertPicture(uint8_t* image_array, uint8_t* new_image_array)
{
    // Allocate the device input nd output vector
    uint8_t* gpuA = NULL;
    cudaMalloc((uint8_t**)&gpuA, N * M * C * sizeof(uint8_t));

    uint8_t* gpuB = NULL;
    cudaMalloc((uint8_t**)&gpuB, N * M * C * sizeof(uint8_t));

    // Copy the host input vector and output int in host memory to the device input
    cudaMemcpy(gpuA, image_array, (N * M * C * sizeof(uint8_t)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, new_image_array, (N * M * C * sizeof(uint8_t)), cudaMemcpyHostToDevice);

    // Execute kernel
    int blocksPerGrid = N * M * C;
    int threadsPerBlock = (1);
    flipImage <<<blocksPerGrid, threadsPerBlock >>> (gpuA, gpuB, N * M * C);

    cudaMemcpy(new_image_array, gpuB, (N * M * C * sizeof(uint8_t)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
}

int main()
{
    // Read the image
    uint8_t* image_array = get_image_array();

    // Allocate output
    uint8_t* new_image_array = (uint8_t*)malloc(M * N * C);

    //invert the picture
    invertPicture(image_array, new_image_array);

    // Save the image
    save_image_array(new_image_array);

    return 0;
}
