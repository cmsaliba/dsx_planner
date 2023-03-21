#include <cfloat>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <helper_math.h>

#define _USE_MATH_DEFINES

#include "kernels.cuh"


// Moller Trumbore algorithm for ray triangle intersection, +1 if no
// intersection, -1 if intersection
__device__ float intersect_triangle(
    float3 orig,
    float3 dir,
    float3 v0,
    float3 v1,
    float3 v2)
{
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;

    // calculate triangle normal vector
    float3 pvec = cross(dir, e2);
    float det = dot(e1, pvec);

    // ray is parallel to the plane
    if (det < FLT_EPSILON && det > -FLT_EPSILON)
    {
        return 1.f;
    }

    float invDet = 1.f / det;
    float3 tvec = orig - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.f || u > 1.f)
    {
        return 1.f;
    }

    float3 qvec = cross(tvec, e1);
    float v = dot(dir, qvec) * invDet;
    if (v < 0.f || u + v > 1.f)
    {
        return 1.f;
    }

    float t = dot(e2, qvec) * invDet;
    if (t < 0.0)
    {
        return 1.f;
    }

    return -1.f;
}

// calculate unsigned distances of the dField grid points
__global__ void
calculateUnsignedDistancesKernel(
    float* dField, 
    float dFieldOffsetX, float dFieldOffsetY, float dFieldOffsetZ,
    float voxelSizeX, float voxelSizeY, float voxelSizeZ,  
    int gridSizeX, int gridSizeY, int gridSizeZ,
    float* meshPts, 
    int nMeshPts, 
    int meshStart, 
    int step)
{
    int gidx = threadIdx.x + blockIdx.x * blockDim.x;
    int gidy = threadIdx.y + blockIdx.y * blockDim.y;
    int gidz = threadIdx.z + blockIdx.z * blockDim.z;

    int gid = gridSizeX * gridSizeY * gidz + gridSizeX * gidy + gidx;

    if (gid < gridSizeX*gridSizeY*gridSizeZ)
    {
        // coordinates of the dField grid point
        float3 fPoint = make_float3(dFieldOffsetX, dFieldOffsetY, dFieldOffsetZ) + make_float3((float)gidx, (float)gidy, (float)gidz)*make_float3(voxelSizeX, voxelSizeY, voxelSizeZ);

        // initialize the minimum distance for the grid point
        float minDist = (meshStart == 0) ? FLT_MAX : dField[gid];

        // check distances to all mesh points in this step
        float tmpDist;
        for(size_t i = meshStart; (i < (meshStart + step)) && i < nMeshPts; i++)
        {
            float3 mPoint = make_float3(meshPts[i * 3], meshPts[i * 3 + 1], meshPts[i * 3 + 2]);
            tmpDist = length(mPoint - fPoint);
            minDist = (tmpDist < minDist) ? tmpDist : minDist;
        }

        // set the dField grid point distance to the new minimum distance
        dField[gid] = minDist;
    }
}

// sign the distances of the dField grid points
__global__ void
signDistancesKernel(
    float* dField, 
    float dFieldOffsetX, float dFieldOffsetY, float dFieldOffsetZ, 
    float voxelSizeX, float voxelSizeY, float voxelSizeZ,  
    int gridSizeX, int gridSizeY, int gridSizeZ,
    float* meshPts, 
    unsigned int* meshFaces, 
    int nMeshFaces, 
    int faceStart, 
    int step)
{
    int gidx = threadIdx.x + blockIdx.x * blockDim.x;
    int gidy = threadIdx.y + blockIdx.y * blockDim.y;
    int gidz = threadIdx.z + blockIdx.z * blockDim.z;

    int gid = gridSizeX * gridSizeY * gidz + gridSizeX * gidy + gidx;

    if (gid < gridSizeX*gridSizeY*gridSizeZ)
    {
        // coordinates of the dField grid point
        float3 fPoint = make_float3(dFieldOffsetX, dFieldOffsetY, dFieldOffsetZ) + make_float3((float)gidx, (float)gidy, (float)gidz)*make_float3(voxelSizeX, voxelSizeY, voxelSizeZ);

        // direction of ray to cast (toward origin)
        float3 dir = -fPoint;

        // determine the sign of the distance by checking for intersection with
        // every face of the mesh - an odd number of intersections means the
        // dfield grid point is inside the mesh and the distance is negative
        float3 v0, v1, v2;
        float sign = 1.f;
        for (size_t i = faceStart; (i < (faceStart + step)) && i < nMeshFaces; i++)
        {
            // indices of the triangle vertices
            unsigned int ind0 = meshFaces[i * 3];
            unsigned int ind1 = meshFaces[i * 3 + 1];
            unsigned int ind2 = meshFaces[i * 3 + 2];

            // get the triangle vertices
            v0.x = meshPts[ind0 * 3];
            v0.y = meshPts[ind0 * 3 + 1];
            v0.z = meshPts[ind0 * 3 + 2];

            v1.x = meshPts[ind1 * 3];
            v1.y = meshPts[ind1 * 3 + 1];
            v1.z = meshPts[ind1 * 3 + 2];

            v2.x = meshPts[ind2 * 3];
            v2.y = meshPts[ind2 * 3 + 1];
            v2.z = meshPts[ind2 * 3 + 2];

            sign *= intersect_triangle(fPoint, dir, v0, v1, v2);
        }

        // set the dField grid point distance to the new minimum distance
        dField[gid] *= sign;
    }
}

// clear drr
__global__ void
clearDRRKernel(
    float* img, int nPix)
{
    // X and Y positions of the pixel in the DRR image
    int gidx = threadIdx.x + blockIdx.x * blockDim.x;
    int gidy = threadIdx.y + blockIdx.y * blockDim.y;

    if (gidx < nPix && gidy < nPix)
    {
        img[nPix * gidy + gidx] = 0;
    }
}

// ray trace
__global__ void
raycastKernel(
    float* dField,
    float dFieldOffsetX, float dFieldOffsetY, float dFieldOffsetZ, 
    float voxelSizeX, float voxelSizeY, float voxelSizeZ,
    int gridSizeX, int gridSizeY, int gridSizeZ,
    float* imv,
    float imPosX, float imPosY, float imPosZ,
    float* aabb,
    float* img,
    float pixSize, int nPix, float weight, float step, float density)
{
    // X and Y positions of the pixel in the DRR image
    int gidx = threadIdx.x + blockIdx.x * blockDim.x;
    int gidy = threadIdx.y + blockIdx.y * blockDim.y;

    if (gidx < nPix && gidy < nPix)
    {
        // coordinates of the pixel and direction of the look ray in source space
        float u = (gidx - (nPix + 1.f) / 2.f) * pixSize;
        float v = (gidy - (nPix + 1.f) / 2.f) * pixSize;
        float3 pixPos = make_float3(u, v, 0.f)
            + make_float3(imPosX, imPosY, imPosZ);
        float3 look = step * normalize(pixPos);

        // ray origin in object space
        float3 rayOrigin = make_float3(imv[12], imv[13], imv[14]);

        // look ray of the pixel in object space
        float3 rayDirection = make_float3(
            dot(make_float3(imv[0], imv[4], imv[8]), look),
            dot(make_float3(imv[1], imv[5], imv[9]), look),
            dot(make_float3(imv[2], imv[6], imv[10]), look));

        // min and max corners of the bounding box
        float3 boxMin = make_float3(aabb[0], aabb[1], aabb[2]);
        float3 boxMax = make_float3(aabb[3], aabb[4], aabb[5]);

        // compute intersection of ray with all six planes of the bounding box
        float3 tBot = (boxMin - rayOrigin) / rayDirection;
        float3 tTop = (boxMax - rayOrigin) / rayDirection;

        // re-order intersections to find smallest and largest on each axis
        float3 tMin = fminf(tTop, tBot);
        float3 tMax = fmaxf(tTop, tBot);

        // find the largest tMin and smallest tMax
        float near = fmaxf(fmaxf(tMin.x, tMin.y), tMin.z);
        float far = fminf(fminf(tMax.x, tMax.y), tMax.z);

        // if ray does not intersect the boundign box, skip it
        if(!(far > near)) return;

        // clamp the ray to the near plane
        if (near < 0.f) near = 0.f;

        // perform ray marching from back to front in uniform steps - step size was
        // set in the look ray

        // start at pixel and march backwards
        float t = fminf(far, length(pixPos)/step);
        // current point on the ray in space
        float3 rayPoint;
        // point in distance field coordinates
        float3 lookup;
        // distance field offset from the origin and voxel size
        float3 dFieldOffset = make_float3(
            dFieldOffsetX, dFieldOffsetY, dFieldOffsetZ);
        float3 dFieldVoxelSize = make_float3(
            voxelSizeX, voxelSizeY, voxelSizeZ);
        // parameters used in the spline interpolation
        float3 i1, i2;
        int i1x, i1y, i1z, i2x, i2y, i2z;
        float tmp1, tmp2;
        float P0, P1, P2, P3, P4, P5, P6, P7;
        float A, B, C, D, E, F;
        // distance from the point on the ray to the surface
        float dist;
        // ray intensity at pixel
        float intensity = 0.f;

        // march!
        while (t>near)
        {
            // point in space and in distance field
            rayPoint = rayOrigin + t*rayDirection;
            lookup = (rayPoint - dFieldOffset) / dFieldVoxelSize;

            // spline interpolation to get distance to surface
            i1 = floorf(lookup);
            i2 = i1 + make_float3(1.f, 1.f, 1.f);

            i1x = (int)i1.x;
            i1y = (int)i1.y;
            i1z = (int)i1.z;

            i2x = (int)i2.x;
            i2y = (int)i2.y;
            i2z = (int)i2.z;

            if (i2x > (gridSizeX-1) || (i2y > gridSizeY-1) || (i2z > gridSizeZ-1) || i1x < 0 || i1y < 0 || i1z < 0)
            {
                intensity += 0.f;
            }
            else
            {
                P0 = dField[i1x + gridSizeX * i1y + gridSizeX*gridSizeY * i1z];
                P1 = dField[i2x + gridSizeX * i1y + gridSizeX*gridSizeY * i1z];
                P2 = dField[i2x + gridSizeX * i1y + gridSizeX*gridSizeY * i2z];
                P3 = dField[i1x + gridSizeX * i1y + gridSizeX*gridSizeY * i2z];
                P4 = dField[i1x + gridSizeX * i2y + gridSizeX*gridSizeY * i1z];
                P5 = dField[i2x + gridSizeX * i2y + gridSizeX*gridSizeY * i1z];
                P6 = dField[i2x + gridSizeX * i2y + gridSizeX*gridSizeY * i2z];
                P7 = dField[i1x + gridSizeX * i2y + gridSizeX*gridSizeY * i2z];

                tmp1 = lookup.x - (float)i1x;
                tmp2 = (float)i2x - lookup.x;

                A = tmp1 * P2 + tmp2 * P3;
                B = tmp1 * P1 + tmp2 * P0;
                C = tmp1 * P5 + tmp2 * P4;
                D = tmp1 * P6 + tmp2 * P7;

                tmp1 = lookup.y - (float)i1y;
                tmp2 = (float)i2y - lookup.y;

                E = tmp1 * D + tmp2 * A;
                F = tmp1 * C + tmp2 * B;

                tmp1 = lookup.z - (float)i1z;
                tmp2 = (float)i2z - lookup.z;

                dist = tmp1 * E + tmp2 * F;

                // If point inside mesh, add to intensity.
                if (dist <= 0.0) intensity += weight;
            }

            t -= 1.f;
        }

        // clamp pixel intensity
        if (intensity > 0.f)
        {
            intensity *= density;
            img[nPix * gidy + gidx] = clamp(
                img[nPix * gidy + gidx] + intensity, 0.f, FLT_MAX);
        }
    }
}

void calculate_unsigned_distances(dim3 dimGrid, dim3 dimBlock, float* d_dField, 
    float* dFieldOffset, float* voxelSize, int* gridSize, float* d_meshPts, 
    int nMeshPts, int meshStart, int step)
{
    calculateUnsignedDistancesKernel <<< dimGrid, dimBlock >>> (d_dField, 
        dFieldOffset[0], dFieldOffset[1], dFieldOffset[2], 
        voxelSize[0], voxelSize[1], voxelSize[2],
        gridSize[0], gridSize[1], gridSize[2],
        d_meshPts, nMeshPts, meshStart, step);
}

void sign_distances(dim3 dimGrid, dim3 dimBlock, float* d_dField, 
    float* dFieldOffset, float* voxelSize, int* gridSize, float* d_meshPts, 
    unsigned int* d_meshFaces, int nMeshFaces, int faceStart, int step)
{
    signDistancesKernel <<< dimGrid, dimBlock >>> (d_dField, 
        dFieldOffset[0], dFieldOffset[1], dFieldOffset[2],
        voxelSize[0], voxelSize[1], voxelSize[2],
        gridSize[0], gridSize[1], gridSize[2],
        d_meshPts, d_meshFaces, nMeshFaces, faceStart, step);
}

void raycast(dim3 dimGrid, dim3 dimBlock, 
    float* d_dField, float* dFieldOffset, float* voxelSize, int* gridSize, 
    float* d_imv, float* imPos, float* d_aabb, float* d_img,
    float pixSize, int nPix, float weight, float step, float density)
{
    raycastKernel <<< dimGrid, dimBlock >>> (d_dField,
        dFieldOffset[0], dFieldOffset[1], dFieldOffset[2],
        voxelSize[0], voxelSize[1], voxelSize[2],
        gridSize[0], gridSize[1], gridSize[2],
        d_imv,
        imPos[0], imPos[1], imPos[2],
        d_aabb,
        d_img,
        pixSize, nPix, weight, step, density);
}

void clearDRR(dim3 dimGrid, dim3 dimBlock, float* img, int nPix)
{
    clearDRRKernel <<< dimGrid, dimBlock >>> (img, nPix);
}