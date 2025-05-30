#ifndef __ANNF_HPP__
#define __ANNF_HPP__

#include <vector>
#include <stack>
#include <limits>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <time.h>
#include <functional>
#include <opencv2/cudaarithm.hpp>

#include "norm2.hpp"
#include "whs.hpp"

/************************* KDTree class *************************/

template <typename ForwardIterator> void
generate_seq(ForwardIterator it, int first, int last)
{
    for (int i = first; i < last; ++i, ++it)
        *it = i;
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

template <typename Tp, int cn> class KDTree
{
private:
    class KDTreeComparator
    {
        const KDTree <Tp, cn> *main; // main class
        int dimIdx; // dimension to compare

    public:
        bool operator () (const int &x, const int &y) const
        {
            cv::Vec <Tp, cn> u = main->data[main->idx[x]];
            cv::Vec <Tp, cn> v = main->data[main->idx[y]];

            return  u[dimIdx] < v[dimIdx];
        }

        KDTreeComparator(const KDTree <Tp, cn> *_main, int _dimIdx)
            : main(_main), dimIdx(_dimIdx) {}
    };

    const int height, width;
    const int leafNumber; // maximum number of point per leaf
    const int zeroThresh; // radius of prohibited shifts

    std::vector <cv::Vec <Tp, cn> > data;
    std::vector <int> idx;
    std::vector <cv::Point2i> nodes;

    int getMaxSpreadN(const int left, const int right) const;
    void operator =(const KDTree <Tp, cn> &) const {};

public:
    void updateDist(const int leaf, const int &idx0, int &bestIdx, double &dist);

    KDTree(const cv::Mat &data, const int leafNumber = 8, const int zeroThresh = 16);
    ~KDTree(){};
};

template <typename Tp, int cn> int KDTree <Tp, cn>::
getMaxSpreadN(const int left, const int right) const
{
    cv::Vec<Tp, cn> maxValue = data[ idx[left] ],
                    minValue = data[ idx[left] ];

    for (int i = left + 1; i < right; ++i)
        for (int j = 0; j < cn; ++j)
        {
            minValue[j] = std::min( minValue[j], data[idx[i]][j] );
            maxValue[j] = std::max( maxValue[j], data[idx[i]][j] );
        }
    cv::Vec<Tp, cn> spread = maxValue - minValue;

    Tp *begIt = &spread[0];
    return int(std::max_element(begIt, begIt + cn) - begIt);
}

template <typename Tp, int cn> KDTree <Tp, cn>::
KDTree(const cv::Mat &img, const int _leafNumber, const int _zeroThresh)
    : height(img.rows), width(img.cols),
      leafNumber(_leafNumber), zeroThresh(_zeroThresh)
///////////////////////////////////////////////////
{
    int imgch = img.channels();
    CV_Assert( img.isContinuous() && imgch <= cn);

    for(size_t i = 0; i < img.total(); i++)
    {
        cv::Vec<Tp, cn> v = cv::Vec<Tp, cn>::all((Tp)0);
        for (int c = 0; c < imgch; c++)
        {
            v[c] = *((Tp*)(img.data) + i*imgch + c);
        }
        data.push_back(v);
    }

    generate_seq( std::back_inserter(idx), 0, int(data.size()) );
    std::fill_n( std::back_inserter(nodes),
        int(data.size()), cv::Point2i(0, 0) );

    std::stack <int> left, right;
    left.push( 0 );
    right.push( int(idx.size()) );

    while ( !left.empty() )
    {
        int  _left = left.top();   left.pop();
        int _right = right.top(); right.pop();

        if ( _right - _left <= leafNumber)
        {
            for (int i = _left; i < _right; ++i)
                nodes[idx[i]] = cv::Point2i(_left, _right);
            continue;
        }

        int nth = _left + (_right - _left)/2;

        int dimIdx = getMaxSpreadN(_left, _right);
        KDTreeComparator comp( this, dimIdx );

        std::vector<int> _idx(idx.begin(), idx.end());
        std::nth_element(/**/
            _idx.begin() +  _left,
            _idx.begin() +    nth,
            _idx.begin() + _right, comp
                         /**/);
        idx = _idx;

          left.push(_left); right.push(nth + 1);
        left.push(nth + 1);  right.push(_right);
    }
}

template <typename Tp, int cn> void KDTree <Tp, cn>::
updateDist(const int leaf, const int &idx0, int &bestIdx, double &dist)
{
    for (int k = nodes[leaf].x; k < nodes[leaf].y; ++k)
    {
        int y = idx0/width, ny = idx[k]/width;
        int x = idx0%width, nx = idx[k]%width;

        if (abs(ny - y) < zeroThresh &&
            abs(nx - x) < zeroThresh)
            continue;
        if (nx >= width  - 1 || nx < 1 ||
            ny >= height - 1 || ny < 1 )
            continue;

        double ndist = norm2(data[idx0], data[idx[k]]);

        if (ndist < dist)
        {
            dist = ndist;
            bestIdx = idx[k];
        }
    }
}

/************************** ANNF search **************************/

static void dominantTransforms(const cv::cuda::GpuMat &img, std::vector <cv::Point2i> &transforms,
                               const int nTransform, const int psize)
{
    const int zeroThresh = 2*psize;
    const int leafNum = 64;

    /** Walsh-Hadamard Transformation **/

    std::vector <cv::cuda::GpuMat> channels;
    cv::cuda::split(img, channels);

    int cncase = std::max(img.channels() - 2, 0);
    //const int np[] = {cncase == 0 ? 12 : (cncase == 1 ? 16 : 10),
    //                  cncase == 0 ? 12 : (cncase == 1 ? 04 : 02),
    //                  cncase == 0 ? 00 : (cncase == 1 ? 04 : 02),
    //                  cncase == 0 ? 00 : (cncase == 1 ? 00 : 10)};

    const int np[] = { cncase == 0 ? 12 : (cncase == 1 ? 4 : 10),
                      cncase == 0 ? 12 : (cncase == 1 ? 1 : 02),
                      cncase == 0 ? 00 : (cncase == 1 ? 1 : 02),
                      cncase == 0 ? 00 : (cncase == 1 ? 00 : 10) };

    int nCh = 0;
    for (int i = 0; i < img.channels(); ++i)
    {
        rgb2whs(channels[i], channels[i], np[i], psize);
        nCh += np[i];
    }

    void* whsptr;
    // allocate unified memory space for result image and assign it to the resultptr
    cudaMallocManaged(&whsptr, img.cols * img.rows * sizeof(float) * nCh);

    cv::cuda::GpuMat whsCuda(img.rows, img.cols, CV_MAKETYPE(CV_32F, nCh), whsptr); // Walsh-Hadamard series
    cv::Mat whs(img.rows, img.cols, CV_MAKETYPE(CV_32F, nCh), whsptr); // Walsh-Hadamard series

    cv::cuda::merge(channels, whsCuda);

//    KDTree <float, 24> kdTree(whs, leafNum, zeroThresh);
    KDTree <float, 6> kdTree(whs, leafNum, zeroThresh);

    std::vector <int> annf( whs.total(), 0 );

    /** Propagation-assisted kd-tree search **/

    for (int i = 0; i < whs.rows; ++i)
        for (int j = 0; j < whs.cols; ++j)
        {
            double dist = std::numeric_limits <double>::max();
            int current = i*whs.cols + j;

            int dy[] = {0, 1, 0}, dx[] = {0, 0, 1};
            for (int k = 0; k < int( sizeof(dy)/sizeof(int) ); ++k)
                if ( i - dy[k] >= 0 && j - dx[k] >= 0 )
                {
                    int neighbor = (i - dy[k])*whs.cols + (j - dx[k]);
                    int leafIdx = (dx[k] == 0 && dy[k] == 0)
                        ? neighbor : annf[neighbor] + dy[k]*whs.cols + dx[k];
                    kdTree.updateDist(leafIdx, current,
                                annf[i*whs.cols + j], dist);
                }
        }

    /** Local maxima extraction **/

    cv::Mat_<double> annfHist(2*whs.rows - 1, 2*whs.cols - 1, 0.0),
                    _annfHist(2*whs.rows - 1, 2*whs.cols - 1, 0.0);
    for (size_t i = 0; i < annf.size(); ++i)
        ++annfHist( annf[i]/whs.cols - int(i)/whs.cols + whs.rows - 1,
                    annf[i]%whs.cols - int(i)%whs.cols + whs.cols - 1 );

    void* whsptr;
    // allocate unified memory space for result image and assign it to the resultptr
    cudaMallocManaged(&whsptr, img.cols * img.rows * sizeof(float) * nCh);

    cv::GaussianBlur( annfHist, annfHist,
        cv::Size(0, 0), std::sqrt(2.0), 0.0, cv::BORDER_CONSTANT);
    cv::dilate( annfHist, _annfHist,
        cv::Matx<uint8_t, 9, 9>::ones() );

    std::vector < std::pair<double, int> > amount;
    std::vector <cv::Point2i> shiftM;

    for (int i = 0, t = 0; i < annfHist.rows; ++i)
    {
        double  *pAnnfHist =  annfHist.ptr<double>(i);
        double *_pAnnfHist = _annfHist.ptr<double>(i);

        for (int j = 0; j < annfHist.cols; ++j)
            if ( pAnnfHist[j] != 0 && pAnnfHist[j] == _pAnnfHist[j] )
            {
                amount.push_back( std::make_pair(pAnnfHist[j], t++) );
                shiftM.push_back( cv::Point2i(j - whs.cols + 1,
                                              i - whs.rows + 1) );
            }
    }

    int num = std::min((int)amount.size(), (int)nTransform);
    std::partial_sort( amount.begin(), amount.begin() + num,
        amount.end(), std::greater< std::pair<double, int> >() );

    transforms.resize(num);
    for (int i = 0; i < num; ++i)
    {
        int idx = amount[i].second;
        transforms[i] = cv::Point2i( shiftM[idx].x, shiftM[idx].y );
    }

    cudaFree(whsptr);
}

#endif /* __ANNF_HPP__ */
