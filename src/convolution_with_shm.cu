//%%cuda --name convolution_with_shm.cu


#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

bool TEST_MODE = false;

// dummy kernel that does nothing
__global__ void warmupKernel(int *dummy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    dummy[i] = 0;
}

// __constant__ float constantKernel[9];

__global__ void convolutionKernel(const float *input, float *output, int width, 
                                    int height, const float *kernel, int kernelSize) {
    int center = (kernelSize - 1) / 2;

    // x is the column index, y is the row index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // fill kernel into shared memory
    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //     for (int i = 0; i < kernelSize * kernelSize; ++i) {
    //         sharedKernel[i] = kernel[i];
    //     }
    // }

    extern __shared__ float sharedKernel[]; 
    int K2 = kernelSize * kernelSize;
    if (threadIdx.x < K2) {
        sharedKernel[threadIdx.x] = kernel[threadIdx.x];
    }

    __syncthreads();

    if (x >= center && x < width - center && y >= center && y < height - center) {
        float sum = 0.0;

        // apply convolution
        for (int ky = 0; ky < kernelSize; ++ky) {
            for (int kx = 0; kx < kernelSize; ++kx) {
                int imageX = x + kx - center;
                int imageY = y + ky - center;

                sum += input[imageY * width + imageX] * sharedKernel[ky * kernelSize + kx];
            }
        }
        // set the output pixel value
        output[y * width + x] = sum;
    }
}

void applyLoGFilterCUDA(const float *input, float *output, int width, int height, int blockDimX, 
                        int blockDimY, int gridDimX, int gridDimY, int gridDimZ, int dim, float& milliseconds) {
    // define the Laplacian of Gaussian (LoG) kernel
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
    
    int threadsPerBlock = blockDimX * blockDimY;
    if (threadsPerBlock > 1024) {
        std::cerr << "Error: The number of threads per block must be less than or equal to 1024." << std::endl;
        milliseconds = -1.0f;
        return;
    }
    // Allocate device memory
    float *d_input, *d_output, *d_kernel;
    gpuErrchk(cudaMalloc((void**)&d_input, width * height * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_output, width * height * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_kernel, kernelSize * kernelSize * sizeof(float)));

    // Copy data from host to device
    gpuErrchk(cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice));

    // Configure and launch the CUDA kernel with shared memory
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim(gridDimX, gridDimY, gridDimZ);
    int sharedMemorySize = kernelSize * kernelSize * sizeof(float);

    // copy kernel to constant memory
    // cudaMemcpyToSymbol(constantKernel, kernel, 9 * sizeof(float));

    // warmup kernel
    int *dummy;
    cudaMalloc((void**)&dummy, width * height * sizeof(int));
    warmupKernel<<<gridDim, blockDim>>>(dummy);
    cudaFree(dummy);

    // create cuda event for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // print number of thread in each block
    std::cout << "blockDimX: " << blockDimX << std::endl;
    std::cout << "blockDimY: " << blockDimY << std::endl;
    std::cout << "gridDimX: " << gridDimX << std::endl;
    std::cout << "gridDimY: " << gridDimY << std::endl;
    std::cout << "gridDimZ: " << gridDimZ << std::endl;
    std::cout << "number of threads in each block: " << blockDimX * blockDimY << std::endl;
    std::cout << "total number of blocks: " << gridDimX * gridDimY * gridDimZ << std::endl;
    std::cout << "total number of threads: " << gridDimX * gridDimY * gridDimZ * blockDimX * blockDimY << std::endl;

    // start timing
    cudaEventRecord(start);
    convolutionKernel<<<gridDim, blockDim, sharedMemorySize>>>(d_input, d_output, width, height, d_kernel, kernelSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "time taken: " << milliseconds << " ms" << std::endl;

    // copy the result back to the host
    cudaMemcpy(output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    // free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    // destroy CUDA event
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void configure_grid_2D(int width, int height, const int& blockDimX, const int& blockDimY, int& gridDimX, int& gridDimY, int& gridDimZ) {
    gridDimX = (width + blockDimX - 1) / blockDimX;
    gridDimY = (height + blockDimY - 1) / blockDimY;
    gridDimZ = 1;
}

void saveOutputImagePNG(const float *outputImage, int width, int height, const std::string& filename) {
    // create output image buffer
    unsigned char *outputImageChar = new unsigned char[width * height];

    // convert output to unsigned char
    for (int i = 0; i < width * height; ++i) {
        outputImageChar[i] = (unsigned char) std::round(outputImage[i] * 255.0f);
    }

    // Save the output image
    stbi_write_png(filename.c_str(), width, height, 1, outputImageChar, 0);

    delete[] outputImageChar;
}

int main(int argc, char *argv[]) {

    // ./program <input_image> <output_image> | ./program <input_image> <output_image> <-test>
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " input_image output_image" << std::endl;
        return 1;
    }
    else if (argc == 4 && std::string(argv[3]) != "-test") {
        std::cerr << "Usage: " << argv[0] << " input_image output_image" << std::endl;
        return 1;
    }

    if (argc == 4) {
        std::cout << "Running in test mode." << std::endl;
        TEST_MODE = true;
    }

    // Load input image
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

    if (TEST_MODE) {
        int blockDimX;
        int blockDimY;
        int gridDimX, gridDimY, gridDimZ;
        float milliseconds = 0.0f;

        int fixedDimensionsX[9] = {32, 64, 128, 256, 512, 1024, 8, 16, 32};
        int fixedDimensionsY[9] = {1, 1, 1, 1, 1, 1, 8, 16, 32};
        int numFixedDimensions = 9;

        // block combination array ex: 1 row as 1 block, 2 rows as 1 block, etc.
        int blockDimXArray[6] = {width, width / 2, width / 4, width/2, width/4, 1};
        int blockDimYArray[6] = {1, 1, 1, 2, 4, height};
        int numBlockCombinations = 6;

        // test fixed dimensions
        for (int i = 0; i < numFixedDimensions; ++i) {
            blockDimX = fixedDimensionsX[i];
            blockDimY = fixedDimensionsY[i];
            std::cout << std::endl << std::endl;

            configure_grid_2D(width, height, blockDimX, blockDimY, gridDimX, gridDimY, gridDimZ);
            std::string filename = "output_2D_" + std::to_string(blockDimX) + "_" + std::to_string(blockDimY) + ".png";
            applyLoGFilterCUDA(inputImageFloat, outputImageFloat, width, height, blockDimX, blockDimY, gridDimX, gridDimY, gridDimZ, 2, milliseconds);
            saveOutputImagePNG(outputImageFloat, width, height, filename);
        }

        // test configurable dimensions
        for (int i = 0; i < numBlockCombinations; ++i) {
            blockDimX = blockDimXArray[i];
            blockDimY = blockDimYArray[i];
            std::cout << std::endl << std::endl;            

            configure_grid_2D(width, height, blockDimX, blockDimY, gridDimX, gridDimY, gridDimZ);

            applyLoGFilterCUDA(inputImageFloat, outputImageFloat, width, height, blockDimX, blockDimY, gridDimX, gridDimY, gridDimZ, 2, milliseconds);
            std::string filename = "output_2D_" + std::to_string(blockDimX) + "_" + std::to_string(blockDimY) + ".png";
            saveOutputImagePNG(outputImageFloat, width, height, filename);
        }
        return 0;
    }

    // apply Laplacian of Gaussian (LoG) filter using CUDA with shared memory

    // one block per row
    int blockDimX = width;
    int blockDimY = 1;
    int gridDimX, gridDimY, gridDimZ;

    configure_grid_2D(width, height, blockDimX, blockDimY, gridDimX, gridDimY, gridDimZ);
    
    float sum = 0.0f;
    float milliseconds = 0.0f;
    int count = 0;
    for (int i = 0; i < 9; ++i) {
        applyLoGFilterCUDA(inputImageFloat, outputImageFloat, width, height, blockDimX, blockDimY, gridDimX, gridDimY, gridDimZ, 2, milliseconds);
        if (milliseconds != -1.0f) {
            ++count;
            sum += milliseconds;
        }
    }

    std::cout << "average time taken: " << sum / count << " ms" << std::endl;

    saveOutputImagePNG(outputImageFloat, width, height, argv[2]);

    // clean up
    stbi_image_free(inputImage);
    delete[] outputImageFloat;

    std::cout << "Laplacian of Gaussian (LoG) edge detection completed successfully." << std::endl;

    return 0;
}
