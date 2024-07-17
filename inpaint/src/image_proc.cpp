#pragma once
#include "image_proc.h"
#include <memory>
void erodeImage(const uint8_t* inputImage, uint8_t* outputImage, const uint8_t* structuringElement, int nW, int nH, int kW, int kH, int nIter)
{
    uint8_t* input = (uint8_t*)malloc(nW * nH * sizeof(uint8_t));
    uint8_t* output = (uint8_t*)malloc(nW * nH * sizeof(uint8_t));
    memcpy(input, inputImage, nW * nH * sizeof(uint8_t));
    // Iterate over the image pixels
    for (int n = 0; n < nIter; n++) {
#pragma omp parallel for
        for (int y = kH / 2; y < nH - kH / 2 - 1; y++) {
            for (int x = kW / 2; x < nW - kW / 2 - 1; x++) {
                bool erode = true; // Flag to check if all structuring element pixels match
                for (int i = 0; i < kH; i++) {
                    for (int j = 0; j < kW; j++) {
                        // Apply the erosion check
                        int posX = x + j - kW / 2;
                        int posY = y + i - kH / 2;

                        // Check boundary conditions
                        if (input[posY * nW + posX] != structuringElement[i * kW + j]) {
                            erode = false;
                            break;
                        }
                    }
                    if (!erode) break;
                }

                // Set the pixel value in the output image based on the erosion result
                output[y * nW + x] = erode ? 255 : 0;
            }
        }
        if (n != nIter - 1) {
            memcpy(input, output, nW * nH * sizeof(uint8_t));
        }        
    }
    memcpy(outputImage, output, nW * nH * sizeof(uint8_t));
    free(input);
    free(output);
}
