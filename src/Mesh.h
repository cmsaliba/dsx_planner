#pragma once

#include <string>

// Structure to represent a 3D mesh.
struct Mesh
{
	unsigned int nVertices;
	unsigned int nFaces;

	// Initialize pointers to nullptr so deletion does not cause an error if
	// data is not loaded.
	unsigned int* faces = nullptr;
	float* allPoints = nullptr;
	float* vertices = nullptr;
	float* centroids = nullptr;
	float* normals = nullptr;
	float* faceAreas = nullptr;
	float aabb[6];
};

int load_mesh_from_file(const std::string& filename, Mesh* mesh);
int save_mesh_to_file(const std::string& filename, Mesh* mesh);
int clear_mesh_data(Mesh* mesh);
