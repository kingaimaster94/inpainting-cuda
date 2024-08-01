#include "inpaint.h"
#include "annf.hpp"
#include "photomontage.hpp"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudalegacy.hpp>
#include <opencv2/cudawarping.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;

void inpaint(const Mat& _src, const Mat& _mask, Mat& dst)
{
    const int nTransform = 60;
    const int psize = 8;
    const cv::Point2i dsize = cv::Point2i(400, 400);

    /** Preparing input **/
    GpuMat _srcGPU, _maskGPU;
    _srcGPU.upload(_src);
    _maskGPU.upload(_mask);
    cv::cuda::cvtColor(_srcGPU, _srcGPU, cv::COLOR_BGR2Lab);

    GpuMat src, mask, imgGpu, dmask, ddmask;    
    cv::cuda::threshold(_maskGPU, _maskGPU, 128, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    const float ls = std::max(/**/ std::min( /*...*/
        std::max(_src.rows, _src.cols) / float(dsize.x),
        std::min(_src.rows, _src.cols) / float(dsize.y)
    ), 1.0f /**/);

    int width = _mask.cols / ls;
    int height = _mask.rows / ls;
    cv::Size scaleSize(width, height);

    cv::cuda::resize(_maskGPU, mask, scaleSize, 0, 0, cv::INTER_NEAREST);
    cv::cuda::resize(_srcGPU, src, scaleSize, 0, 0, cv::INTER_AREA);

    _srcGPU.convertTo(imgGpu, CV_32F);
    imgGpu.setTo(0, mask);

    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));

    auto erode = cv::cuda::createMorphologyFilter(MORPH_ERODE, mask.type(), element, Point(-1, -1), 2);
    erode->apply(mask, dmask);
    erode->apply(dmask, ddmask);

    std::vector <Point2i> pPath;
    cv::Mat_<int> backref(ddmask.size(), int(-1));

    for (int i = 0; i < ddmask.rows; ++i)
    {
        uint8_t* dmask_data = (uint8_t*)ddmask.template ptr<uint8_t>(i);
        int* backref_data = (int*)backref.template ptr< int >(i);

        for (int j = 0; j < ddmask.cols; ++j)
            if (dmask_data[j] == 0)
            {
                backref_data[j] = int(pPath.size());
                pPath.push_back(cv::Point(j, i));
            }
    }

    /** ANNF computation **/
    std::vector <cv::Point2i> transforms(nTransform);
    dominantTransforms(imgGpu, transforms, nTransform, psize);
    transforms.push_back(cv::Point2i(0, 0));

    /** Warping **/
    std::vector<std::vector<cv::Vec<float, 3>>> pointSeq(pPath.size()); // source image transformed with transforms
    std::vector<int> labelSeq(pPath.size());                                 // resulting label sequence
    std::vector<std::vector<int>>  linkIdx(pPath.size());                  // neighbor links for pointSeq elements
    std::vector<std::vector<unsigned char>> maskSeq(pPath.size());        // corresponding mask

    for (size_t i = 0; i < pPath.size(); ++i)
    {
        uint8_t xmask = dmask.template at<uint8_t>(pPath[i]);

        for (int j = 0; j < nTransform + 1; ++j)
        {
            cv::Point2i u = pPath[i] + transforms[j];

            unsigned char vmask = 0;
            cv::Vec <float, 3> vimgGpu = 0;

            if (u.y < src.rows && u.y >= 0
                && u.x < src.cols && u.x >= 0)
            {
                if (xmask == 0 || j == nTransform)
                    vmask = mask.template at<uint8_t>(u);
                vimgGpu = imgGpu.template at<cv::Vec<float, 3> >(u);
            }

            maskSeq[i].push_back(vmask);
            pointSeq[i].push_back(vimgGpu);

            if (vmask != 0)
                labelSeq[i] = j;
        }

        cv::Point2i  p[] = {
                             pPath[i] + cv::Point2i(0, +1),
                             pPath[i] + cv::Point2i(+1, 0)
        };

        for (uint j = 0; j < sizeof(p) / sizeof(cv::Point2i); ++j)
            if (p[j].y < src.rows && p[j].y >= 0 &&
                p[j].x < src.cols && p[j].x >= 0)
                linkIdx[i].push_back(backref(p[j]));
            else
                linkIdx[i].push_back(-1);
    }

    /** Stitching **/
    photomontage(pointSeq, maskSeq, linkIdx, labelSeq);

    /** Upscaling **/
    if (ls != 1)
    {
        _src.convertTo(imgGpu, CV_32F);

        std::vector <Point2i> __pPath = pPath; pPath.clear();

        cv::Mat_<int> __backref(imgGpu.size(), -1);

        std::vector <std::vector <cv::Vec <float, 3> > > __pointSeq = pointSeq; pointSeq.clear();
        std::vector <int> __labelSeq = labelSeq; labelSeq.clear();
        std::vector <std::vector <int> > __linkIdx = linkIdx; linkIdx.clear();
        std::vector <std::vector <unsigned char > > __maskSeq = maskSeq; maskSeq.clear();

        for (size_t i = 0; i < __pPath.size(); ++i)
        {
            cv::Point2i p[] = {
                __pPath[i] + cv::Point2i(0, -1),
                __pPath[i] + cv::Point2i(-1, 0)
            };

            for (uint j = 0; j < sizeof(p) / sizeof(cv::Point2i); ++j)
                if (p[j].y < src.rows && p[j].y >= 0 &&
                    p[j].x < src.cols && p[j].x >= 0)
                    __linkIdx[i].push_back(backref(p[j]));
                else
                    __linkIdx[i].push_back(-1);
        }

        for (size_t k = 0; k < __pPath.size(); ++k)
        {
            int clabel = __labelSeq[k];
            int nearSeam = 0;

            for (size_t i = 0; i < __linkIdx[k].size(); ++i)
                nearSeam |= (__linkIdx[k][i] == -1
                    || clabel != __labelSeq[__linkIdx[k][i]]);

            if (nearSeam != 0)
                for (int i = 0; i < ls; ++i)
                    for (int j = 0; j < ls; ++j)
                    {
                        cv::Point2i u = ls * (__pPath[k] + transforms[__labelSeq[k]]) + cv::Point2i(j, i);

                        pPath.push_back(ls * __pPath[k] + cv::Point2i(j, i));
                        labelSeq.push_back(0);

                        __backref(i, j) = int(pPath.size());

                        cv::Point2i dv[] = {
                                             cv::Point2i(0,  0),
                                             cv::Point2i(-1, 0),
                                             cv::Point2i(+1, 0),
                                             cv::Point2i(0, -1),
                                             cv::Point2i(0, +1)
                        };

                        std::vector <cv::Vec <float, 3> > pointVec;
                        std::vector <uint8_t> maskVec;

                        for (uint q = 0; q < sizeof(dv) / sizeof(cv::Point2i); ++q)
                            if (u.x + dv[q].x >= 0 && u.x + dv[q].x < imgGpu.cols
                                && u.y + dv[q].y >= 0 && u.y + dv[q].y < imgGpu.rows)
                            {
                                pointVec.push_back(imgGpu.template at<cv::Vec <float, 3> >(u + dv[q]));
                                maskVec.push_back(_mask.template at<uint8_t>(u + dv[q]));
                            }
                            else
                            {
                                pointVec.push_back(cv::Vec <float, 3>::all(0));
                                maskVec.push_back(0);
                            }

                        pointSeq.push_back(pointVec);
                        maskSeq.push_back(maskVec);
                    }
            else
            {
                cv::Point2i fromIdx = ls * (__pPath[k] + transforms[__labelSeq[k]]),
                    toIdx = ls * __pPath[k];

                for (int i = 0; i < ls; ++i)
                {
                    cv::Vec <float, 3>* from = imgGpu.template ptr<cv::Vec <float, 3> >(fromIdx.y + i) + fromIdx.x;
                    cv::Vec <float, 3>* to = imgGpu.template ptr<cv::Vec <float, 3> >(toIdx.y + i) + toIdx.x;

                    for (int j = 0; j < ls; ++j)
                        to[j] = from[j];
                }
            }
        }


        for (size_t i = 0; i < pPath.size(); ++i)
        {
            cv::Point2i  p[] = {
                pPath[i] + cv::Point2i(0, +1),
                pPath[i] + cv::Point2i(+1, 0)
            };

            std::vector <int> linkVec;

            for (uint j = 0; j < sizeof(p) / sizeof(cv::Point2i); ++j)
                if (p[j].y < src.rows && p[j].y >= 0 &&
                    p[j].x < src.cols && p[j].x >= 0)
                    linkVec.push_back(__backref(p[j]));
                else
                    linkVec.push_back(-1);

            linkIdx.push_back(linkVec);
        }

        photomontage(pointSeq, maskSeq, linkIdx, labelSeq);
    }

    /** Writing result **/
    for (size_t i = 0; i < labelSeq.size(); ++i)
    {
        if (pPath[i].x >= imgGpu.cols || pPath[i].y >= imgGpu.rows)
            continue;

        cv::Vec <float, 3> val = pointSeq[i][labelSeq[i]];
        imgGpu.template at<cv::Vec <float, 3> >(pPath[i]) = val;
    }
    imgGpu.convertTo(dst, dst.type());
    cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);
}

