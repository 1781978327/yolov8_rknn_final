#include <thread>
#include <iostream>
#include "deepsort.h"
#include "mytime.h"
#include "track.h"

using namespace std;

DeepSort::DeepSort(std::string modelPath, int batchSize, int featureDim, int cpu_id, rknn_core_mask npu_id) {
    this->npu_id = npu_id;
    this->cpu_id = cpu_id;
    this->enginePath = modelPath;
    this->batchSize = batchSize;
    this->featureDim = featureDim;
    this->imgShape = cv::Size(128, 256);
    this->maxBudget = 100;
    this->maxCosineDist = 0.5;
    init();
}

void DeepSort::init() {
    objTracker = new tracker(maxCosineDist, maxBudget);
    featureExtractor1 = new FeatureTensor(enginePath.c_str(), cpu_id, npu_id, 1, 1);
    featureExtractor1->init(imgShape, featureDim, NET_INPUTCHANNEL);
    featureExtractor2 = new FeatureTensor(enginePath.c_str(), cpu_id, npu_id, 1, 1);
    featureExtractor2->init(imgShape, featureDim, NET_INPUTCHANNEL);
}

DeepSort::~DeepSort() {
    delete objTracker;
}

void DeepSort::sort(cv::Mat& frame, vector<DetectBox>& dets) {
    DETECTIONS detections;
    vector<CLSCONF> clsConf;
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.confidence));
    }

    result.clear();
    results.clear();
    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }

    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b; b.x1 = i(0); b.y1 = i(1); b.x2 = i(2)+i(0); b.y2 = i(3)+i(1); b.confidence = 1.;
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].confidence = c.conf;
    }
}

void DeepSort::sort(cv::Mat& frame, DETECTIONS& detections) {
    bool flag = featureExtractor1->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}

void DeepSort::sort_interval(cv::Mat& frame, vector<DetectBox>& dets) {
    result.clear();
    results.clear();
    objTracker->predict();
    for (Track& track : objTracker->tracks) {
        if (!track.is_confirmed() || track.time_since_update > this->track_interval + 1)
            continue;
        result.push_back(make_pair(track.track_id, track.to_tlwh()));
        results.push_back(make_pair(CLSCONF(track.cls, track.conf), track.to_tlwh()));
    }
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b; b.x1 = i(0); b.y1 = i(1); b.x2 = i(2)+i(0); b.y2 = i(3)+i(1); b.confidence = 1.;
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].confidence = c.conf;
    }
}

void DeepSort::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;

    int numOfDetections = detections.size();
    bool flag1 = true, flag2 = true;
    if (numOfDetections < 2) {
        flag1 = featureExtractor1->getRectsFeature(frame, detections);
        flag2 = true;
    } else {
        DETECTIONS detectionsPart1, detectionsPart2;
        int border = numOfDetections >> 1;
        auto start = detections.begin(), end = detections.end();
        detectionsPart1.assign(start, start + border);
        detectionsPart2.assign(start + border, end);

        thread reID1Thread1 (&FeatureTensor::getRectsFeature, featureExtractor1, std::ref(frame), std::ref(detectionsPart1));
        thread reID1Thread2 (&FeatureTensor::getRectsFeature, featureExtractor2, std::ref(frame), std::ref(detectionsPart2));
        reID1Thread1.join(); reID1Thread2.join();

        for (int idx = 0; flag1 && flag2 && idx < numOfDetections; idx++) {
            if (idx < border)
                detections[idx].updateFeature(detectionsPart1[idx].feature);
            else
                detections[idx].updateFeature(detectionsPart2[idx - border].feature);
        }
    }

    if (flag1 && flag2) {
        objTracker->predict();
        objTracker->update(detectionsv2);
        result.clear();
        results.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf), track.to_tlwh()));
        }
    }
}

int DeepSort::track_process(){
    cout << "Warning: track_process() is deprecated. Use sort() method directly." << endl;
    return 0;
}

void DeepSort::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes) {
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 2);
        std::string lbl = cv::format("ID:%d", (int)box.trackID);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0));
    }
    cv::imwrite("./display.jpg", temp);
}

std::vector<Track> DeepSort::get_confirmed_tracks() const {
    std::vector<Track> confirmed;
    for (const Track& t : objTracker->tracks) {
        if (t.is_confirmed())
            confirmed.push_back(t);
    }
    return confirmed;
}
