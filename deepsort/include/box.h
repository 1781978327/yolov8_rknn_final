#ifndef DEEPSORT_BOX_H
#define DEEPSORT_BOX_H

#ifndef DETECTBOX_DEFINED
typedef struct {
    float x1, y1, x2, y2;
    float confidence;
    float classID;
    float trackID;
} DetectBox;
#endif

#endif // DEEPSORT_BOX_H
