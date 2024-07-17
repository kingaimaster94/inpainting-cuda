#ifndef __PHOTOMONTAGE_HPP__
#define __PHOTOMONTAGE_HPP__

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

namespace gcoptimization
{

#include "gcgraph.hpp"


typedef float TWeight;
typedef  int  labelTp;


#define GCInfinity 10*1000*1000
#define eps 0.02


template <typename Tp> static int min_idx(std::vector <Tp> vec)
{
    return int( std::min_element(vec.begin(), vec.end()) - vec.begin() );
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

template <typename Tp> class Photomontage
{
private:
    const std::vector <std::vector <Tp> > &pointSeq;   // points for stitching
    const std::vector <std::vector <uint8_t> > &maskSeq; // corresponding masks

    const std::vector <std::vector <int> > &linkIdx;   // vector of neighbors for pointSeq

    std::vector <std::vector <labelTp> > labelings;    // vector of labelings
    std::vector <TWeight>  distances;                  // vector of max-flow costs for different labeling

    std::vector <labelTp> &labelSeq;                   // current best labeling

    TWeight singleExpansion(const int alpha);          // single neighbor computing

    void operator =(const Photomontage <Tp>&) const {};

protected:
    virtual TWeight dist(const Tp &l1p1, const Tp &l1p2, const Tp &l2p1, const Tp &l2p2);
    virtual void setWeights(GCGraph <TWeight> &graph,
        const int idx1, const int idx2, const int l1, const int l2, const int lx);

public:
    void gradientDescent(); // gradient descent in alpha-expansion topology

    Photomontage(const std::vector <std::vector <Tp> > &pointSeq,
                 const std::vector <std::vector <uint8_t> > &maskSeq,
                 const std::vector <std::vector <int> > &linkIdx,
                       std::vector <labelTp> &labelSeq);
    virtual ~Photomontage(){};
};

template <typename Tp> inline TWeight Photomontage <Tp>::
dist(const Tp &l1p1, const Tp &l1p2, const Tp &l2p1, const Tp &l2p2)
{
    return norm2(l1p1, l2p1) + norm2(l1p2, l2p2);
}

template <typename Tp> void Photomontage <Tp>::
setWeights(GCGraph <TWeight> &graph, const int idx1, const int idx2,
    const int l1, const int l2, const int lx)
{
    if ((size_t)idx1 >= pointSeq.size() || (size_t)idx2 >= pointSeq.size()
        || (size_t)l1 >= pointSeq[idx1].size() || (size_t)l1 >= pointSeq[idx2].size()
        || (size_t)l2 >= pointSeq[idx1].size() || (size_t)l2 >= pointSeq[idx2].size()
        || (size_t)lx >= pointSeq[idx1].size() || (size_t)lx >= pointSeq[idx2].size())
        return;

    if (l1 == l2)
    {
        /** Link from A to B **/
        TWeight weightAB = dist( pointSeq[idx1][l1], pointSeq[idx2][l1],
                                 pointSeq[idx1][lx], pointSeq[idx2][lx] );
        graph.addEdges( idx1, idx2, weightAB, weightAB );
    }
    else
    {
        int X = graph.addVtx();

        /** Link from X to sink **/
        TWeight weightXS = dist( pointSeq[idx1][l1], pointSeq[idx2][l1],
                                 pointSeq[idx1][l2], pointSeq[idx2][l2] );
        graph.addTermWeights( X, 0, weightXS );

        /** Link from A to X **/
        TWeight weightAX = dist( pointSeq[idx1][l1], pointSeq[idx2][l1],
                                 pointSeq[idx1][lx], pointSeq[idx2][lx] );
        graph.addEdges( idx1, X, weightAX, weightAX );

        /** Link from X to B **/
        TWeight weightXB = dist( pointSeq[idx1][lx], pointSeq[idx1][lx],
                                 pointSeq[idx1][l2], pointSeq[idx1][l2] );
        graph.addEdges( X, idx2, weightXB, weightXB );
    }
}

template <typename Tp> TWeight Photomontage <Tp>::
singleExpansion(const int alpha)
{
    GCGraph <TWeight> graph( 3*int(pointSeq.size()), 4*int(pointSeq.size()) );

    /** Terminal links **/
    for (size_t i = 0; i < maskSeq.size(); ++i)
        graph.addTermWeights( graph.addVtx(),
            maskSeq[i][alpha] ? TWeight(0) : TWeight(GCInfinity), 0 );

    /** Neighbor links **/
    for (size_t i = 0; i < pointSeq.size(); ++i)
        for (size_t j = 0; j < linkIdx[i].size(); ++j)
            if ( linkIdx[i][j] != -1)
                setWeights( graph, int(i), linkIdx[i][j],
                    labelSeq[i], labelSeq[linkIdx[i][j]], alpha );

    /** Max-flow computation **/
    TWeight result = graph.maxFlow();

    /** Writing results **/
    for (size_t i = 0; i < pointSeq.size(); ++i)
        labelings[i][alpha] = graph.inSourceSegment(int(i)) ? labelSeq[i] : alpha;

    return result;
}

template <typename Tp> void Photomontage <Tp>::
gradientDescent()
{
    TWeight optValue = std::numeric_limits<TWeight>::max();

    for (int num = -1; /**/; num = -1)
    {
        int range = int( pointSeq[0].size() );
#pragma omp parallel for
        for (int i = 0; i < range; i++) {
            distances[i] = singleExpansion(i);
        }

        int minIndex = min_idx(distances);
        TWeight minValue = distances[minIndex];

        if (minValue < (1.00 - eps) * optValue)
        {
            optValue = distances[num = minIndex];
        }

        if (num == -1)
        {
            break;
        }

#pragma omp parallel for
        for (int i = 0; i < labelSeq.size(); ++i)
        {
            labelSeq[i] = labelings[i][num];
        }
    }
}

template <typename Tp> Photomontage <Tp>::
Photomontage( const std::vector <std::vector <Tp> > &_pointSeq,
            const std::vector <std::vector <uint8_t> > &_maskSeq,
              const std::vector <std::vector <int> > &_linkIdx,
                              std::vector <labelTp> &_labelSeq )
  :
    pointSeq(_pointSeq), maskSeq(_maskSeq), linkIdx(_linkIdx),
    distances(pointSeq[0].size()), labelSeq(_labelSeq)
{
    size_t lsize = pointSeq[0].size();
    labelings.assign( pointSeq.size(),
      std::vector <labelTp>( lsize ) );
}

}

template <typename Tp> static inline
void photomontage( const std::vector <std::vector <Tp> > &pointSeq,
                 const std::vector <std::vector <uint8_t> > &maskSeq,
                   const std::vector <std::vector <int> > &linkIdx,
                   std::vector <gcoptimization::labelTp> &labelSeq )
{
    gcoptimization::Photomontage <Tp>(pointSeq, maskSeq,
        linkIdx, labelSeq).gradientDescent();
}

#endif /* __PHOTOMONTAGE_HPP__ */
