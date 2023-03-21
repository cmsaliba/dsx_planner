#include "Mesh.h"

#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>

int load_mesh_from_file(const std::string& filename, Mesh* mesh)
{
	std::ifstream file(filename, std::ios::in | std::ios::binary);

	size_t headInfo_size = 2 * sizeof(unsigned int);
	char* headInfo_char = new char[headInfo_size];
	file.read(headInfo_char, headInfo_size);
	unsigned int* headInfo = (unsigned int*)headInfo_char;
	mesh->nVertices = headInfo[0];
	mesh->nFaces = headInfo[1];

	size_t aabb_size = 6 * sizeof(float);
	char* aabb_char = new char[aabb_size];
	file.read(aabb_char, aabb_size);
	std::memcpy(mesh->aabb, aabb_char, aabb_size);

	size_t allPoints_size = (mesh->nVertices + mesh->nFaces) * 3 * sizeof(float);
	char* allPoints_char = new char[allPoints_size];
	file.read(allPoints_char, allPoints_size);
	mesh->allPoints = new float[mesh->nVertices * 3 + mesh->nFaces * 3];
	std::memcpy(mesh->allPoints, allPoints_char, allPoints_size);
	mesh->vertices = &mesh->allPoints[0];
	mesh->centroids = &mesh->allPoints[mesh->nVertices * 3];

	size_t normals_size = mesh->nVertices * 3 * sizeof(float);
	char* normals_char = new char[normals_size];
	file.read(normals_char, normals_size);
	mesh->normals = new float[mesh->nVertices * 3];
	std::memcpy(mesh->normals, normals_char, normals_size);

	size_t faces_size = mesh->nFaces * 3 * sizeof(unsigned int);
	char* faces_char = new char[faces_size];
	file.read(faces_char, faces_size);
	mesh->faces = new unsigned int[mesh->nFaces * 3];
	std::memcpy(mesh->faces, faces_char, faces_size);

	size_t faceAreas_size = mesh->nFaces * sizeof(float);
	char* faceAreas_char = new char[faceAreas_size];
	file.read(faceAreas_char, faceAreas_size);
	mesh->faceAreas = new float[mesh->nFaces];
	std::memcpy(mesh->faceAreas, faceAreas_char, faceAreas_size);

	file.close();

	delete[] headInfo_char;
	delete[] aabb_char;
	delete[] allPoints_char;
	delete[] normals_char;
	delete[] faces_char;
	delete[] faceAreas_char;

	return 0;
}

int save_mesh_to_file(const std::string& filename, Mesh* mesh)
{
	size_t headInfo_size = 2 * sizeof(unsigned int);
	unsigned int headInfo[2] = {mesh->nVertices, mesh->nFaces};
	char* headInfoMemblock = new char[headInfo_size];
	std::memcpy(headInfoMemblock, headInfo, headInfo_size);

	size_t aabb_size = 6 * sizeof(float);
	char* aabbMemblock = new char[aabb_size];
	std::memcpy(aabbMemblock, mesh->aabb, aabb_size);

	size_t allPoints_size = (mesh->nVertices + mesh->nFaces) * 3 * sizeof(float);
	char* allPointsMemblock = new char[allPoints_size];
	std::memcpy(allPointsMemblock, mesh->allPoints, allPoints_size);

	size_t normals_size = mesh->nVertices * 3 * sizeof(float);
	char* normalsMemblock = new char[normals_size];
	std::memcpy(normalsMemblock, mesh->normals, normals_size);

	size_t faces_size = mesh->nFaces * 3 * sizeof(unsigned int);
	char* facesMemblock = new char[faces_size];
	std::memcpy(facesMemblock, mesh->faces, faces_size);

	size_t faceAreas_size = mesh->nFaces * sizeof(float);
	char* faceAreasMemblock = new char[faceAreas_size];
	std::memcpy(faceAreasMemblock, mesh->faceAreas, faceAreas_size);

	std::ofstream file(filename, std::ios::out | std::ios::binary);

	file.write(headInfoMemblock, headInfo_size);
	file.write(aabbMemblock, aabb_size);
	file.write(allPointsMemblock, allPoints_size);
	file.write(normalsMemblock, normals_size);
	file.write(facesMemblock, faces_size);
	file.write(faceAreasMemblock, faceAreas_size);

	file.close();

	delete[] headInfoMemblock;
	delete[] aabbMemblock;
	delete[] allPointsMemblock;
	delete[] normalsMemblock;
	delete[] facesMemblock;
	delete[] faceAreasMemblock;

	return 0;
}

int clear_mesh_data(Mesh* mesh)
{
	delete[] mesh->allPoints;
	delete[] mesh->normals;
	delete[] mesh->faces;
	delete[] mesh->faceAreas;

	mesh->allPoints = nullptr;
	mesh->normals = nullptr;
	mesh->faces = nullptr;
	mesh->faceAreas = nullptr;
	mesh->vertices = nullptr;
	mesh->centroids = nullptr;

	return 0;
}
