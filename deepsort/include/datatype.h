#ifndef DEEPSORTDATATYPE_H
#define DEEPSORTDATATYPE_H

#include <cstddef>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

typedef struct CLSCONF {
    CLSCONF() {
        this->cls = -1;
        this->conf = -1;
    }
    CLSCONF(int cls, float conf) {
        this->cls = cls;
        this->conf = conf;
    }
    int cls;
    float conf;
} CLSCONF;

typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> DETECTBOX_TLWH;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;
typedef DETECTBOX_TLWH DETECTBOX;  // alias for compatibility
typedef Eigen::Matrix<float, 1, 512, Eigen::RowMajor> FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 512, Eigen::RowMajor> FEATURESS;

typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

using RESULT_DATA = std::pair<int, DETECTBOX>;
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
typedef struct t{
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
} TRACHER_MATCHD;

typedef Eigen::Matrix<float, -1, -1, Eigen::RowMajor> DYNAMICM;

#endif // DEEPSORTDATATYPE_H
