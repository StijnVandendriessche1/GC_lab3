
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <fstream>

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 941       // VR width
#define N 704       // VR height
#define C 3         // Colors
#define OFFSET 15   // Header length

//writeToCSV
//help function to write stuff to csv
void writeRecordToFile(std::string filename, int fieldOne, int fieldTwo, float fieldThree)
{
    std::ofstream file;
    file.open(filename, std::ios_base::app);
    file << fieldOne << "," << fieldTwo << "," << fieldThree << std::endl;
    file.close();
}

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

//cpu function that inverts image
void invertPictureCPU(uint8_t* input, uint8_t* output)
{
    for (int i = 0; i < N * M * C; i++)
    {
        output[i] = 255 - input[i];
    }
}

//invert kernel GPU
__global__ void flipImage(uint8_t* input, uint8_t* output, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        output[i] = 255 - input[i];
    }
}

//function that inverts image using gpu
void invertPictureGPU(uint8_t* image_array, uint8_t* new_image_array)
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
    int blocksPerGrid = ceil((N*M*C)/ threadsPerBlock);
    flipImage <<<blocksPerGrid, threadsPerBlock >>> (gpuA, gpuB, N * M * C);

    cudaMemcpy(new_image_array, gpuB, (N * M * C * sizeof(uint8_t)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
}

// STRIDE PRAMETERS
// parameters are chosen specificly for my GPU
// NVIDIA GTX 2080 Super with max-Q Design
// 48 SM units
// 3072 cores
// 64 cores per SM unit
// 2 warps per SM unit
// this means that the optimal number of threads per block is 32 or 64 (respectively 1 or 2 warps)
// to use the whole GPU, a total of 48 blocks should be used when using 64 threads per block
// or 96 blocks when using 32 threads per block
// since all of the cores will be used, a stride of 3072 is used in the kernel

//invert kernel GPU stride
__global__ void flipImageStride(uint8_t* input, uint8_t* output, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // + 3072 every iteration
    bool done = false;
    while (!done)
    {
        if (i < numElements)
        {
            output[i] = 255 - input[i];
            i += 3072;
        }
        else
        {
            done = true;
        }
    }
}

//function that inverts image using gpu and strinding 
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
    // int blocksPerGrid = N * M * C;
    // int threadsPerBlock = (1);
    int threadsPerBlock = 64;
    //int blocksPerGrid = ceil((N*M*C)/ threadsPerBlock);
    int blocksPerGrid = 48;
    flipImageStride << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuB, N * M * C);

    cudaMemcpy(new_image_array, gpuB, (N * M * C * sizeof(uint8_t)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
}

//time both normal GPU and GPU stride functions
void getTimeInversionGPU(uint8_t* input, uint8_t* outputGPU, uint8_t* outputStride, float* times)
{
    //run the kernel once to avoid startup delay
    invertPictureGPU(input, outputGPU);

    cudaEvent_t startGPU, startStride, stopGPU, stopStride;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&startStride);
    cudaEventCreate(&stopGPU);
    cudaEventCreate(&stopStride);

    //time gpu
    cudaEventRecord(startGPU);
    invertPictureGPU(input, outputGPU);
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);
    float millisGPU;
    cudaEventElapsedTime(&millisGPU, startGPU, stopGPU);

    //time stride
    cudaEventRecord(startStride);
    invertPictureStride(input, outputStride);
    cudaEventRecord(stopStride);
    cudaEventSynchronize(stopStride);
    float millisStride;
    cudaEventElapsedTime(&millisStride, startStride, stopStride);

    times[0] = millisGPU;
    times[1] = millisStride;
}

//invert kernel GPU stride
__global__ void flipImageStrideLoop(uint8_t* input, uint8_t* output, int numElements, int numThreads)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // + 3072 every iteration
    while (true)
    {
        if (i < numElements)
        {
            output[i] = 255 - input[i];
            i += numThreads;
        }
        else
        {
            return;
        }
    }
}

//function that inverts image using gpu and strinding on different 
void invertPictureStrideLoop(uint8_t* image_array, uint8_t* new_image_array, int blocks, int threads)
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
    // int blocksPerGrid = N * M * C;
    // int threadsPerBlock = (1);
    //int threadsPerBlock = 64;
    //int blocksPerGrid = ceil((N*M*C)/ threadsPerBlock);
    //int blocksPerGrid = 48;
    for (int i = 0; i < 10; i++)
    {
        flipImageStrideLoop << <blocks, threads >> > (gpuA, gpuB, N * M * C, threads * blocks);
    }

    cudaMemcpy(new_image_array, gpuB, (N * M * C * sizeof(uint8_t)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
}

//time gpu loop function
float timedLoopInversion(uint8_t* image_array, uint8_t* new_image_array, int blocks, int theads)
{
    cudaEvent_t startGPU, startStride, stopGPU, stopStride;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&startStride);
    cudaEventCreate(&stopGPU);
    cudaEventCreate(&stopStride);

    //time gpu
    cudaEventRecord(startGPU);
    invertPictureStrideLoop(image_array, new_image_array, blocks, theads);
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);
    float millis;
    cudaEventElapsedTime(&millis, startGPU, stopGPU);

    return millis/20;
}

//loop function for stride loop
void StrideLoop(uint8_t* image_array, uint8_t* new_image_array)
{
    for (int i = 2; i <= 32; i+=2)
    {
        for (int j = 2; j <= 32; j+=2)
        {
            //invertPictureStrideLoop(image_array, new_image_array, i, j);
            float time = timedLoopInversion(image_array, new_image_array, i, j);
            //printf("amount of blocks:\t%d\tamount of threads:\t%d\ttime:\t%f\n", i,j,time);
            writeRecordToFile("strideTimesBlocksavg32.csv", i, j, time);
        }
    }
}


//red kernel
__global__ void flipImageRed(uint8_t* input, uint8_t* output, int numElements, int numThreads)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // + 3072 every iteration
    while (i < (numElements/3))
    {
        output[(3 * i)] = 255 - input[(3 * i)];
        output[(3*i) + 1] = 255 - input[(3 * i) + 1];
        output[(3 * i) + 2] = 255 - input[(3 * i) + 2];

        if (output[(3 * i)] > 100 && output[(3 * i)] < 200)
        {
            output[(3 * i)] = (output[(3 * i)] % 25) * 10;
        }
        i += numThreads;
    }
}

//function that inverts image and alters red value
void invertRed(uint8_t* image_array, uint8_t* new_image_array)
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
    // int blocksPerGrid = N * M * C;
    // int threadsPerBlock = (1);
    int threadsPerBlock = 64;
    //int blocksPerGrid = ceil((N*M*C)/ threadsPerBlock);
    int blocksPerGrid = 48;
    flipImageRed << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuB, N * M * C, threadsPerBlock * blocksPerGrid);

    cudaMemcpy(new_image_array, gpuB, (N * M * C * sizeof(uint8_t)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
}

int main()
{
    // Read the image
    uint8_t* image_array = get_image_array();

    // Allocate memory for output CPU, GPU and stride
    //uint8_t* cpuOut = (uint8_t*)malloc(M * N * C);
    uint8_t* gpuOut = (uint8_t*)malloc(M * N * C);
    //uint8_t* strideOut = (uint8_t*)malloc(M * N * C);

    //invert using cpu, gpu and strid
    //invertPictureCPU(image_array, cpuOut);
    //invertPictureGPU(image_array, gpuOut);
    //invertPictureStride(image_array, strideOut);

    //time the inversions
    //float* times = (float*)malloc(2);
    //getTimeInversionGPU(image_array, gpuOut, strideOut, times);
    invertRed(image_array, gpuOut);
    //printf("GPU time:    %f \nSTRIDE time: %f", times[0], times[1]);

    //StrideLoop(image_array, strideOut);

    // Save the images
    //save_image_array(cpuOut, "./cpu_inverted.ppm");
    save_image_array(gpuOut, "./gpu_inverted_red.ppm");
    //save_image_array(strideOut, "./stride_inverted.ppm");

    return 0;
}
