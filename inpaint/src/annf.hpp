
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

#include "norm2.hpp"

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

static void dominantTransforms(const cv::Mat &img, std::vector <cv::Point2i> &transforms,
                               const int nTransform, const int psize)
{
    const int zeroThresh = 2 * psize;
    const int leafNum = 64;

    int width = img.cols;
    int height = img.rows;
    KDTree <float, 3> kdTree(img, leafNum, zeroThresh);
    std::vector <int> annf(img.total(), 0 );

    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            double dist = std::numeric_limits <double>::max();
            int current = i * width + j;

            int dy[] = { 0, 1, 0 }, dx[] = { 0, 0, 1 };
            for (int k = 0; k < 3; ++k)
            {
                if (i - dy[k] >= 0 && j - dx[k] >= 0)
                {
                    int neighbor = (i - dy[k]) * width + (j - dx[k]);
                    int leafIdx = (dx[k] == 0 && dy[k] == 0) ? neighbor : annf[neighbor] + dy[k] * width + dx[k];
                    kdTree.updateDist(leafIdx, current, annf[i * width + j], dist);
                }
            }
        }
    }

    /** Local maxima extraction **/

    cv::Mat_<double> annfHist(2* height - 1, 2* width - 1, 0.0),
                    _annfHist(2* height - 1, 2* width - 1, 0.0);
#pragma omp parallel for
    for (int i = 0; i < annf.size(); ++i)
    {
        ++annfHist(annf[i] / width - i / width + height - 1, annf[i] % width - int(i) % width + width - 1);
    }

    cv::GaussianBlur( annfHist, annfHist,
        cv::Size(0, 0), std::sqrt(2.0), 0.0, cv::BORDER_CONSTANT);
    cv::dilate( annfHist, _annfHist, cv::Matx<uint8_t, 9, 9>::ones());

    std::vector < std::pair<double, int> > amount;
    std::vector <cv::Point2i> shiftM;

    int t = 0;
    for (int i = 0; i < annfHist.rows; i++) {
        double* pAnnfHist = annfHist.ptr<double>(i);
        double* _pAnnfHist = _annfHist.ptr<double>(i);

        for (int j = 0; j < annfHist.cols; ++j) {
            if (pAnnfHist[j] != 0 && pAnnfHist[j] == _pAnnfHist[j]) {
                amount.push_back(std::make_pair(pAnnfHist[j], t));
                shiftM.push_back(cv::Point2i(j - width + 1, i - height + 1));
                t++;
            }
        }
    }

    int num = std::min((int)amount.size(), (int)nTransform);
    std::partial_sort( amount.begin(), amount.begin() + num,
        amount.end(), std::greater< std::pair<double, int> >() );

    transforms.resize(num);
#pragma omp parallel for
    for (int i = 0; i < num; ++i)
    {
        int idx = amount[i].second;
        transforms[i] = cv::Point2i( shiftM[idx].x, shiftM[idx].y );
    }
}

#endif /* __ANNF_HPP__ */
