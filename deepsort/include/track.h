#ifndef TRACK_H
#define TRACK_H

#include "MyKalmanFilter.h"
#include "datatype.h"
#include "model.hpp"
#include <vector>
#include <utility>

class Track
{
public:
    enum TrackState {Tentative = 1, Confirmed, Deleted};
    Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id,
          int n_init, int max_age, const FEATURE& feature);
    Track(KAL_MEAN& mean, KAL_COVA& covariance, int track_id,
          int n_init, int max_age, const FEATURE& feature, int cls, float conf);
    void predit(MyKalmanFilter* kf);
    void update(MyKalmanFilter* const kf, const DETECTION_ROW &detection);
    void update(MyKalmanFilter* const kf, const DETECTION_ROW & detection, CLSCONF pair_det);
    void mark_missed();
    bool is_confirmed() const;
    bool is_deleted() const;
    bool is_tentative() const;
    DETECTBOX to_tlwh() const;

    void append_trajectory(float cx, float cy);
    const std::vector<std::pair<float, float>>& get_trajectory() const;
    void clear_trajectory();

    static const int MAX_TRAJECTORY_LEN = 50;

    int time_since_update;
    int track_id;
    FEATURESS features;
    KAL_MEAN mean;
    KAL_COVA covariance;
    int hits;
    int age;
    int _n_init;
    int _max_age;
    TrackState state;
    int cls;
    float conf;
    std::vector<std::pair<float, float>> trajectory;
private:
    void featuresAppendOne(const FEATURE& f);
};

#endif // TRACK_H
