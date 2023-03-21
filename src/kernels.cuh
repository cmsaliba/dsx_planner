void calculate_unsigned_distances(dim3 dimGrid, dim3 dimBlock, float* d_dField, 
    float* dFieldOffset, float* voxelSize, int* gridSize, float* d_meshPts, 
    int nMeshPts, int meshStart, int step);

void sign_distances(dim3 dimGrid, dim3 dimBlock, float* d_dField, 
    float* dFieldOffset, float* voxelSize, int* gridSize, float* d_meshPts, 
    unsigned int* d_meshFaces, int nMeshFaces, int faceStart, int step);

void raycast(dim3 dimGrid, dim3 dimBlock, 
    float* d_dField, float* dFieldOffset, float* voxelSize, int* gridSize, 
    float* d_imv, float* imPos, float* d_aabb, float* d_img,
    float pixSize, int nPix, float weight, float step, float density);

void clearDRR(dim3 dimGrid, dim3 dimBlock, float* img, int nPix);