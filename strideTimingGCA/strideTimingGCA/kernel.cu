
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

//HELPER FUNCTIONS
//read the input image
uint8_t* get_image_array(void)
{
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

//save an array of uint8_t as an image
void save_image_array(uint8_t* image_array, char* filename)
{
    // Try opening the file
    FILE* imageFile;
    imageFile = fopen(filename, "wb");
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
//END HELPER FUNCTIONS

//GPU CODE
//invert kernel GPU with stride
__global__ void flipImageStride(uint8_t* input, uint8_t* output, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // + 3072 every iteration
    while (true)
    {
        if (i < numElements)
        {
            output[i] = 255 - input[i];
            i += 3072;
        }
        else
        {
            return;
        }
    }
}

//function that takes care of all the needed steps to execute the stride kernel
void invertPictureStride(uint8_t* image_array, uint8_t* new_image_array)
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
    int threadsPerBlock = 64;
    int blocksPerGrid = 48;
    flipImageStride << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuB, N * M * C);

    cudaMemcpy(new_image_array, gpuB, (N * M * C * sizeof(uint8_t)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
}

//function that executes the GPU code while taking time measurements
float getTimeInversionGPU(uint8_t* input, uint8_t* output)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //time stride
    cudaEventRecord(start);
    invertPictureStride(input, output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float millisStride;
    cudaEventElapsedTime(&millisStride, start, stop);

    //set time
    return millisStride;
}
//END GPU CODE

int main()
{
    // Read the image
    uint8_t* image_array = get_image_array();

    // Allocate memory for output CPU, GPU and stride
    uint8_t* out = (uint8_t*)malloc(M * N * C);

    //time the inversions
    float time = getTimeInversionGPU(image_array, out);
    printf("GPU time:    %f \n", time);

    // Save the images
    save_image_array(out, "./test.ppm");
    return 0;
}