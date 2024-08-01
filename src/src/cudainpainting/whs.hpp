#ifndef __WHS_HPP__
#define __WHS_HPP__
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime.h>

static inline int hl(int x)
{
    int res = 0;
    while (x)
    {
        res += x&1;
        x >>= 1;
    }
    return res;
}

static inline int rp2(int x)
{
    int res = 1;
    while (res < x)
        res <<= 1;
    return res;
}

template <typename ForwardIterator>
static void generate_snake(ForwardIterator snake, const int n)
{
    cv::Point previous;
    if (n > 0)
    {
        previous = cv::Point(0, 0);
        *snake = previous;
    }

    for (int k = 1, num = 1; num <= n; ++k)
    {
        const cv::Point2i dv[] = { cv::Point2i( !(k&1),   (k&1) ),
                                   cv::Point2i( -(k&1), -!(k&1) ) };

        *snake = previous = previous - dv[1];
        ++num;

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < k && num < n; ++j)
            {
                *snake = previous = previous + dv[i];
                ++num;
            }
    }
}

static void nextProjection(std::vector <cv::cuda::GpuMat> &projections, const cv::Point &A,
                           const cv::Point &B, const int psize)
{
    int xsign = (A.x != B.x)*(hl(A.x&B.x) + (B.x > A.x))&1;
    int ysign = (A.y != B.y)*(hl(A.y&B.y) + (B.y > A.y))&1;
    bool plusToMinusUpdate = xsign || ysign;

    int dx = (A.x != B.x) << ( hl(psize - 1) - hl(A.x ^ B.x) );
    int dy = (A.y != B.y) << ( hl(psize - 1) - hl(A.y ^ B.y) );

    cv::cuda::GpuMat proj = projections[projections.size() - 1];

    void* unified_ptr;
    cudaMallocManaged(&unified_ptr, proj.rows * proj.cols * sizeof(float));

    cv::cuda::GpuMat nproj(proj.rows, proj.cols, CV_32FC1, unified_ptr), zeroMat(proj.rows, proj.cols, CV_32FC1, cv::Scalar(0));
    cv::cuda::subtract(zeroMat, proj, nproj);

    for (int i = dy; i < nproj.rows; ++i)
    {
        float *vxNext = nproj.ptr<float>(i - dy);
        float  *vNext = nproj.ptr<float>(i);

        float *vxCurrent = proj.ptr<float>(i - dy);

        if (plusToMinusUpdate)
            for (int j = dx; j < nproj.cols; ++j)
                vNext[j] += vxCurrent[j - dx] - vxNext[j - dx];
        else
            for (int j = dx; j < nproj.cols; ++j)
                vNext[j] -= vxCurrent[j - dx] - vxNext[j - dx];
    }
    projections.push_back(nproj);
}

static void rgb2whs(const cv::cuda::GpuMat &src, cv::cuda::GpuMat &dst, const int nProjections, const int psize)
{
    CV_Assert(nProjections <= psize*psize && src.type() == CV_32FC1);

    const int npsize = rp2(psize);
    std::vector <cv::cuda::GpuMat> projections;

    cv::cuda::GpuMat img, proj;
    cv::cuda::copyMakeBorder(src, img, npsize, npsize, npsize, npsize,
        cv::BORDER_CONSTANT, 0);

    auto boxFilter = cv::cuda::createBoxFilter(CV_32FC3, CV_32FC3, cv::Size(npsize, npsize));
    boxFilter->apply(img, proj);

    projections.push_back(proj);

    std::vector <cv::Point2i> snake_idx;
    generate_snake(std::back_inserter(snake_idx), nProjections);

    for (int i = 1; i < nProjections; ++i)
        nextProjection(projections, snake_idx[i - 1],
            snake_idx[i], npsize);

    int pad = 0;

    cv::cuda::merge(projections, img);
    img(cv::Rect(npsize + pad, npsize + pad, src.cols - pad,
        src.rows - pad)).copyTo(dst);
}


#endif /* __WHS_HPP__ */
