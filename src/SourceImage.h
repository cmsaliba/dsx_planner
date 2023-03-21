#pragma once

#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda_runtime.h>
#include "helper_cuda_gl.h"

class SourceImage
{
public:
	// default constructor
	SourceImage() : 
			cudaTexBuffer(0),
			texBufferSize(0),
			vaoPlane(0),
	  		vertexBufferIDPlane(0),
			uvBufferIDPlane(0),
			faceBufferIDPlane(0),
			vaoSource(0),
			vertexBufferIDSource(0),
			faceBufferIDSource(0),
			normalBufferIDSource(0){}

	// destructor
	~SourceImage();

	void setup_gl();
	void setup_cuda();
	void draw_plane();
	void draw_source();

	glm::mat4 src2world, src2world_saved;
	glm::vec3 imcenterSrc, imcenterSrc_saved;
	float width;
	int nPix;
	float pixSize;

	glm::mat4 planeMVP;
	glm::mat4 sourceMVP;

	GLuint texture;

	glm::vec3 color;

	struct cudaGraphicsResource* cudaTexResource;
	cudaArray* cudaTexArray;
	float* cudaTexBuffer;
	size_t texBufferSize;

	float selectRad;
	bool selected;

private:
	GLuint vaoPlane;
	GLuint vertexBufferIDPlane;
	GLuint uvBufferIDPlane;
	GLuint faceBufferIDPlane;

	GLuint vaoSource;
	GLuint vertexBufferIDSource;
	GLuint faceBufferIDSource;
	GLuint normalBufferIDSource;
};
