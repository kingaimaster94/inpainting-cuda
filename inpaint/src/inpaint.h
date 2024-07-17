#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
void inpaint(const Mat& _src, const Mat& _mask, Mat& res);