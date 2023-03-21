#pragma once

#include <stddef.h>

struct DistanceField
{
    size_t dFieldBufferSize;
    int nPts[3];
    float dFieldSize[3];
    float voxelSize[3];
    float offset[3];

    // float pointer to data - initialize as nullptr so deletion does not cause
    // an error if data is not loaded
    float* dField = nullptr;
};

int clear_dfield_data(DistanceField* distanceField);