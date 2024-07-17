#pragma once
#include "image_proc.h"
#include <memory>

__global__ void erodeImageKernel(const uint8_t* input, uint8_t* output, const uint8_t* structuringElement, int nW, int nH, int kW, int kH, int nIter) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= kW / 2 && x < nW - kW / 2 - 1 && y >= kH / 2 && y < nH - kH / 2 - 1) {
        for (int n = 0; n < nIter; n++) {
            bool erode = true;
            for (int i = 0; i < kH; i++) {
                for (int j = 0; j < kW; j++) {
                    int posX = x + j - kW / 2;
                    int posY = y + i - kH / 2;
                    if (input[posY * nW + posX] != structuringElement[i * kW + j]) {
                        erode = false;
                        break;
                    }
                }
                if (!erode) break;
            }
            output[y * nW + x] = erode ? 255 : 0;
            __syncthreads();
            if (n != nIter - 1) {
                input = output;
                __syncthreads();
            }
        }
    }
}

void erodeImage(const uint8_t* inputImage, uint8_t* outputImage, const uint8_t* structuringElement, int nW, int nH, int kW, int kH, int nIter) {
    uint8_t* d_input, *d_output, *d_structuringElement;

    cudaMalloc((void**)&d_input, nW * nH * sizeof(uint8_t));
    cudaMalloc((void**)&d_output, nW * nH * sizeof(uint8_t));
    cudaMalloc((void**)&d_structuringElement, kW * kH * sizeof(uint8_t));

    cudaMemcpy(d_input, inputImage, nW * nH * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_structuringElement, structuringElement, kW * kH * sizeof(uint8_t), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);  // Adjust the block size as per your requirements
    dim3 gridSize((nW + blockSize.x - 1) / blockSize.x, (nH + blockSize.y - 1) / blockSize.y);

    erodeImageKernel <<<gridSize, blockSize>>> (d_input, d_output, d_structuringElement, nW, nH, kW, kH, nIter);

    cudaMemcpy(outputImage, d_output, nW * nH * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_structuringElement);
}