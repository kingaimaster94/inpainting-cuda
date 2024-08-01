#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// CUDA Kernel for erosion
__global__ void erodeKernel(const uint8_t* src, uint8_t* dst, int width, int height, int kernelRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint8_t minVal = 255;

        for (int dy = -kernelRadius; dy <= kernelRadius; ++dy) {
            for (int dx = -kernelRadius; dx <= kernelRadius; ++dx) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    uint8_t val = src[ny * width + nx];
                    minVal = min(minVal, val);
                }
            }
        }

        dst[y * width + x] = minVal;
    }
}

void erodeImageCUDA(const cv::GpuMat& src, cv::GpuMat& dst, int kernelRadius) {
    // Check if the input image is single-channel
    CV_Assert(src.type() == CV_8UC1);

    int width = src.cols;
    int height = src.rows;

    // Allocate memory on the device
    uint8_t* d_src = src.data;
    uint8_t* d_dst = dst.data;
    cudaMemcpy(d_dst, d_src, width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

    // Define block and grid sizes
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);



    // Launch the erosion kernel
    erodeKernel << <grid, block >> > (d_src, d_dst, width, height, kernelRadius);
 }
