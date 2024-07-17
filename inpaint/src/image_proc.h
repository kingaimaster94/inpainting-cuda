#pragma once
#include <stdint.h>
void erodeImage(const uint8_t* inputImage, uint8_t* outputImage, const uint8_t* structuringElement, int nW, int nH, int kW, int kH, int nIter = 1);