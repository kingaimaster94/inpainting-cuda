#include "inpaint.h"
#include "annf.hpp"
#include "photomontage.hpp"
#include "image_proc.h"

using namespace std;

void inpaint(const Mat& _src, const Mat& _mask, Mat& res)
{
    const int nTransform = 40, psize = 8;
    const cv::Point2i dsize = cv::Point2i(500, 500);
    cv::Mat src, mask, img;

    const float ls = std::max(/**/ std::min( /*...*/
        std::max(_src.rows, _src.cols) / float(dsize.x),
        std::min(_src.rows, _src.cols) / float(dsize.y)
    ), 1.0f /**/);

    int width = _mask.cols / ls;
    int height = _mask.rows / ls;
    cv::Size scaleSize(width, height);
    if (ls != 1) {
        cv::resize(_mask, mask, scaleSize, 0, 0, cv::INTER_NEAREST);
        cv::resize(_src, src, scaleSize, 0, 0, cv::INTER_AREA);
    }
    else {
        mask = _mask;
        src = _src;
    }

    src.convertTo(img, CV_32F);
    img.setTo(0, ~(mask > 0));

    uint8_t* dmask = (uint8_t*)malloc(width * height);
    uint8_t* ddmask = (uint8_t*)malloc(width * height);

    unsigned char structureElement[9];
    memset(structureElement, 0xff, 9 * sizeof(char));

    erodeImage(mask.data, dmask, structureElement, width, height, 3, 3, 2);
    erodeImage(dmask, ddmask, structureElement, width, height, 3, 3, 2);

    std::vector <Point2i> pPath;

    int* backref = (int*)malloc(width * height * sizeof(int));
    memset(backref, 0xffffffff, width * height * sizeof(int));
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            if (ddmask[i * width + j] == 0)
            {
                backref[i * width + j] = int(pPath.size());
                pPath.push_back(cv::Point(j, i));
            }
        }
    }

    /** ANNF computation **/
    cv::TickMeter tm;
    tm.start();
    std::vector <cv::Point2i> transforms(nTransform);
    dominantTransforms(img, transforms, nTransform, psize);
    transforms.push_back(cv::Point2i(0, 0));
    tm.stop();
    printf("dominantTransforms = %f\n", tm.getTimeMilli());

    /** Warping **/
    std::vector <std::vector <cv::Vec <float, 3> > > pointSeq(pPath.size()); // source image transformed with transforms
    std::vector <int> labelSeq(pPath.size());                                 // resulting label sequence
    std::vector <std::vector <int> >  linkIdx(pPath.size());                  // neighbor links for pointSeq elements
    std::vector <std::vector <unsigned char > > maskSeq(pPath.size());        // corresponding mask

    for (int i = 0; i < pPath.size(); ++i)
    {
        for (int j = 0; j < nTransform + 1; ++j)
        {
            cv::Point2i u = pPath[i] + transforms[j];

            uint8_t vmask = 0;
            cv::Vec <float, 3> vimg = 0;

            if (u.y < height && u.y >= 0  && u.x < width && u.x >= 0)
            {
                if (dmask[pPath[i].y * width + pPath[i].x] == 0 || j == nTransform)
                {
                    vmask = mask.template at<uint8_t>(u);
                }
                vimg = img.template at<cv::Vec<float, 3> >(u);
            }

            maskSeq[i].push_back(vmask);
            pointSeq[i].push_back(vimg);

            if (vmask != 0)
            {
                labelSeq[i] = j;
            }
        }

        for (int j = 0; j < 2; ++j) {
            int x = pPath[i].x + j;
            int y = pPath[i].y + (1 - j);
            if (y < height && y >= 0 && x < width && x >= 0)
            {
                linkIdx[i].push_back(backref[y * width + x]);
            }
            else
            {
                linkIdx[i].push_back(-1);
            }
        }
    }

    /** Stitching **/
    tm.reset();
    tm.start();
    photomontage(pointSeq, maskSeq, linkIdx, labelSeq);
    tm.stop();
    printf("photomontage = %f\n", tm.getTimeMilli());

    /** Upscaling **/
    if (ls != 1)
    {
        _src.convertTo(img, CV_32F);

        std::vector <Point2i> __pPath = pPath;
        pPath.clear();

        int width1 = img.cols;
        int height1 = img.rows;
        int* __backref = (int*)malloc(width1 * height1 * sizeof(int));
        memset(__backref, 0xffffffff, width1 * height1 * sizeof(int));


        std::vector <std::vector <cv::Vec <float, 3> > > __pointSeq = pointSeq;
        pointSeq.clear();
        std::vector <int> __labelSeq = labelSeq;
        labelSeq.clear();
        std::vector <std::vector <int> > __linkIdx = linkIdx;
        linkIdx.clear();
        std::vector <std::vector <unsigned char > > __maskSeq = maskSeq;
        maskSeq.clear();

        for (int i = 0; i < __pPath.size(); ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                int x = __pPath[i].x - j;
                int y = __pPath[i].y - (1 - j);
                if (y < height && y >= 0 && x < width && x >= 0)
                {
                    __linkIdx[i].push_back(backref[y * width + x]);
                }
                else
                {
                    __linkIdx[i].push_back(-1);
                }
            }
        }

        for (int k = 0; k < __pPath.size(); ++k)
        {
            int clabel = __labelSeq[k];
            int nearSeam = 0;

            for (size_t i = 0; i < __linkIdx[k].size(); ++i)
            {
                nearSeam |= (__linkIdx[k][i] == -1 || clabel != __labelSeq[__linkIdx[k][i]]);
            }

            if (nearSeam != 0)
            {
                for (int i = 0; i < ls; ++i)
                {
                    for (int j = 0; j < ls; ++j)
                    {
                        cv::Point2i u = ls * (__pPath[k] + transforms[__labelSeq[k]]) + cv::Point2i(j, i);

                        pPath.push_back(ls * __pPath[k] + cv::Point2i(j, i));
                        labelSeq.push_back(0);

                        __backref[i * width1 + j] = int(pPath.size());

                        cv::Point2i dv[] = {
                                             cv::Point2i(0,  0),
                                             cv::Point2i(-1, 0),
                                             cv::Point2i(+1, 0),
                                             cv::Point2i(0, -1),
                                             cv::Point2i(0, +1)
                        };

                        std::vector <cv::Vec <float, 3> > pointVec;
                        std::vector <uint8_t> maskVec;

                        for (uint q = 0; q < 5; ++q)
                        {
                            int x = u.x + dv[q].x;
                            int y = u.y + dv[q].y;
                            if (x >= 0 && x < width1 && y >= 0 && y < height1)
                            {
                                pointVec.push_back(img.template at<cv::Vec <float, 3> >(u + dv[q]));
                                maskVec.push_back(_mask.template at<uint8_t>(u + dv[q]));
                            }
                            else
                            {
                                pointVec.push_back(cv::Vec <float, 3>::all(0));
                                maskVec.push_back(0);
                            }
                        }

                        pointSeq.push_back(pointVec);
                        maskSeq.push_back(maskVec);
                    }
                }
            }
            else
            {
                cv::Point2i fromIdx = ls * (__pPath[k] + transforms[__labelSeq[k]]),
                    toIdx = ls * __pPath[k];

                for (int i = 0; i < ls; ++i)
                {
                    cv::Vec <float, 3>* from = img.template ptr<cv::Vec <float, 3> >(fromIdx.y + i) + fromIdx.x;
                    cv::Vec <float, 3>* to = img.template ptr<cv::Vec <float, 3> >(toIdx.y + i) + toIdx.x;

                    for (int j = 0; j < ls; ++j)
                    {
                        to[j] = from[j];
                    }
                }
            }
        }


        for (size_t i = 0; i < pPath.size(); ++i)
        {
            std::vector <int> linkVec;

            for (uint j = 0; j < 2; ++j)
            {
                int x = pPath[i].x + j;
                int y = pPath[i].y + (1 - j);

                if (y < height && y >= 0 && x < width && x >= 0)
                    linkVec.push_back(__backref[y * width1 + x]);
                else
                    linkVec.push_back(-1);
            }

            linkIdx.push_back(linkVec);
        }

        photomontage(pointSeq, maskSeq, linkIdx, labelSeq);
        free(__backref);
    }
    
    /** Writing result **/
    for (size_t i = 0; i < labelSeq.size(); ++i)
    {
        if (pPath[i].x >= width || pPath[i].y >= height)
            continue;

        cv::Vec <float, 3> val = pointSeq[i][labelSeq[i]];
        img.template at<cv::Vec <float, 3> >(pPath[i]) = val;
    }
    free(dmask);
    free(ddmask);
    free(backref);
    img.convertTo(res, _src.type());
}
