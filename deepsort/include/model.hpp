#ifndef MODEL_HPP
#define MODEL_HPP

#include <algorithm>
#include "datatype.h"

const float kRatio = 0.5;
enum DETECTBOX_IDX { IDX_X = 0, IDX_Y, IDX_W, IDX_H };

class DETECTION_ROW {
public:
    DETECTBOX tlwh;
    float confidence;
    FEATURE feature;

    DETECTBOX to_xyah() const {
        DETECTBOX ret = tlwh;
        ret(0, IDX_X) += (ret(0, IDX_W) * kRatio);
        ret(0, IDX_Y) += (ret(0, IDX_H) * kRatio);
        ret(0, IDX_W) /= ret(0, IDX_H);
        return ret;
    }
    DETECTBOX to_tlbr() const {
        DETECTBOX ret = tlwh;
        ret(0, IDX_X) += ret(0, IDX_W);
        ret(0, IDX_Y) += ret(0, IDX_H);
        return ret;
    }
    void updateFeature(FEATURE& feature_) {
        this->feature = feature_;
    }
};

typedef std::vector<DETECTION_ROW> DETECTIONS;
typedef std::pair<std::vector<CLSCONF>, DETECTIONS> DETECTIONSV2;

#endif // MODEL_HPP
