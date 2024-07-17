#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
#ifdef INPAINTING_EXPORTS
#    define INPAINTIG_API __declspec(dllexport)
#else
#    define INPAINTIG_API __declspec(dllimport)
#endif

INPAINTIG_API void inpaint(const Mat& _src, const Mat& _mask, Mat& res);