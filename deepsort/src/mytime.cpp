#include "mytime.h"
double what_time_is_it_now()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec * 1000 + (double)time.tv_usec * .001;
}
