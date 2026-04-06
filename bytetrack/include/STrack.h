#pragma once

#include <opencv2/opencv.hpp>
#include <deque>
#include "kalmanFilter.h"

using namespace std;

enum TrackState { New = 0, Tracked, Lost, Removed };

typedef struct _CENTER_XY
{
	int x;
	int y;
} CENTER_XY;

class STrack
{
public:
	STrack(vector<float> tlwh_, float score, string name, int label);
	~STrack();

	vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
	void static multi_predict(vector<STrack*> &stracks, byte_kalman::KalmanFilter &kalman_filter);
	void static_tlwh();
	void static_tlbr();
	vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);
	vector<float> to_xyah();
	void mark_lost();
	void mark_removed();
	int next_id();
	int end_frame();

	void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);
	void re_activate(STrack &new_track, int frame_id, bool new_id = false);
	void update(STrack &new_track, int frame_id);

	static void set_max_trail_length(int frames) { max_trail_length = frames; }
	static int get_max_trail_length() { return max_trail_length; }

public:
	bool is_activated;
	int track_id;
	int state;

	vector<float> _tlwh;
	vector<float> tlwh;
	vector<float> tlbr;
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;

	string name;
	int label;

	CENTER_XY last_center;
	CENTER_XY now_center;
	double speed_1s;
	double speed_1m;
	double speed_all;
	double liveness;

	std::deque<std::pair<float,float>> trajectory;
	const std::deque<std::pair<float,float>>& get_trajectory() const { return trajectory; }

private:
	byte_kalman::KalmanFilter kalman_filter;
	static int max_trail_length;
};
