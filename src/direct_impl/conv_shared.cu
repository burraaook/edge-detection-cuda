// %%cuda --name conv_shm.cu

#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool TEST_MODE = false;

// dummy kernel that does nothing
__global__ void warmupKernel(int *dummy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dummy[i] = 0;
}

__global__ void convolutionKernelShared(const float *input, float *output, int width, int height, const float *kernel, int kernelSize) {

    int iy = blockIdx.x + (kernelSize - 1) / 2;

    int ix = threadIdx.x + (kernelSize - 1) / 2;

    // center of kernel in both dimensions
    int center = (kernelSize - 1) / 2;

    int idx = iy * width + ix;

    int threadId = threadIdx.x;
    int K2 = kernelSize * kernelSize;

    // shared memory for the kernel
    extern __shared__ float sdata[];

    // load kernel into shared memory
    if (threadId < K2) {
        sdata[threadId] = kernel[threadId];
    }

    __syncthreads();

    float sum = 0.0;

    // check if the thread is within the image
    if (idx < width * height) {
        for (int ki = 0; ki < kernelSize; ++ki) {
            for (int kj = 0; kj < kernelSize; ++kj) {
                int imageX = ix + kj - center;
                int imageY = iy + ki - center;

                sum += input[imageY * width + imageX] * sdata[ki * kernelSize + kj];
            }
        }

        output[idx] = sum;
    }
}

void applyLoGFilterCUDA(const float *input, float *output, int width, int height, float& milliseconds) {
    // define the  kernel
    float kernel[9] = {
        0, 1, 0,
        1, -4, 1,
        0, 1, 0
    };
    int kernelSize = 3;

    // define 5x5 kernel
    // float kernel[25] = {
    //     0, 0, 1, 0, 0,
    //     0, 1, 2, 1, 0,
    //     1, 2, -16, 2, 1,
    //     0, 1, 2, 1, 0,
    //     0, 0, 1, 0, 0
    // };
    // int kernelSize = 5;
    
    
    // allocate device memory
    float *d_input, *d_output, *d_kernel;
    cudaMalloc((void**)&d_input, width * height * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernelSize * kernelSize * sizeof(float));

    // copy data from host to device
    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // configure and launch the CUDA kernel with shared memory
    int numBlocks = height - kernelSize + 1;
    int threadsPerBlock = width - kernelSize + 1;
    std::cout << "numBlocks: " << numBlocks << std::endl;
    std::cout << "threadsPerBlock: " << threadsPerBlock << std::endl;

    // warmup kernel
    int *dummy;
    cudaMalloc((void**)&dummy, numBlocks * threadsPerBlock * sizeof(int));
    warmupKernel<<<numBlocks, threadsPerBlock>>>(dummy);
    cudaFree(dummy);

    // create cuda event for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start timing
    cudaEventRecord(start);
    convolutionKernelShared<<<numBlocks, threadsPerBlock, kernelSize * kernelSize * sizeof(float)>>>(d_input, d_output, width, height, d_kernel, kernelSize);
    cudaEventRecord(stop);

    // stop timing
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    // copy the result back to the host
    cudaMemcpy(output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

int main(int argc, char *argv[]) {

    // ./program <input_image> <output_image> | ./program <input_image> <output_image> <-test>
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " input_image output_image" << std::endl;
        return 1;
    }

    // load input image
    int width, height, channels;
    unsigned char *inputImage = stbi_load(argv[1], &width, &height, &channels, 1);
    float *outputImageFloat = new float[width * height];

    if (!inputImage) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << std::endl;
        return 1;
    }

    // print width and height
    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;

    // convert input to float
    float *inputImageFloat = new float[width * height];
    for (int i = 0; i < width * height; ++i) {
        inputImageFloat[i] = inputImage[i] / 255.0f;
    }

    float sum = 0.0f;
    float milliseconds = 0.0f;
    for (int i = 0; i < 10; ++i) {
        // apply Laplacian of Gaussian (LoG) filter using CUDA with shared memory
        applyLoGFilterCUDA(inputImageFloat, outputImageFloat, width, height, milliseconds);
        sum += milliseconds;
    }
    milliseconds = sum / 10.0f;
    std::cout << "Average time taken: " << milliseconds << " ms" << std::endl;

    // apply Laplacian of Gaussian (LoG) filter using CUDA with shared memory
    // applyLoGFilterCUDA(inputImageFloat, outputImageFloat, width, height, milliseconds);

    // create output image buffer
    unsigned char *outputImage = new unsigned char[width * height];

    // convert output to unsigned char
    for (int i = 0; i < width * height; ++i) {
        outputImage[i] = (unsigned char) std::round(outputImageFloat[i] * 255.0f);
    }

    // save the output image
    stbi_write_png(argv[2], width, height, 1, outputImage, 0);

    // clean up
    stbi_image_free(inputImage);
    delete[] outputImage;
    delete[] outputImageFloat;

    std::cout << "Laplacian of Gaussian (LoG) edge detection completed successfully." << std::endl;

    return 0;
}
