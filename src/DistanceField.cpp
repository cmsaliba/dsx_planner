#include <cuda_runtime.h>       // runtime library
#include "helper_cuda.h"

#include "DistanceField.h"

int clear_dfield_data(DistanceField* distanceField)
{
    checkCudaErrors(
        cudaFree(distanceField->dField));
    distanceField->dField = nullptr;

    return 1;
}