#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda_runtime.h>       // runtime library
#include <cuda_gl_interop.h>    // interoperability with OpenGL

#include "helper_cuda.h"
#include "helper_cuda_gl.h"

#include "glUtils.h"

#include "SourceImage.h"

// todo: delete texture, cuda resources and arrays
// todo: write free functions for cuda OpenGL and call from destructor

SourceImage::~SourceImage()
{
	// free the device buffer allocated by CUDA - cudaTexBuffer is allocated as 0
	// so cudaFree won't throw an error if it hasn't been allocated
	cudaFree(this->cudaTexBuffer);
}

void SourceImage::setup_gl()
{
	// Source.

	selected = false;
	selectRad = 0.05;
	color = glm::vec3(0.f, 1.f, 0.0f);

	// Source pyramid vertices in source space.
	float sourceVertices[] = {
		0.f, 0.f, 0.f,			// source
		-0.05, 0.05, -0.05,		// top left
		0.05, 0.05, -0.05,		// top right
		0.05, -0.05, -0.05,		// bottom right
		-0.05, -0.05, -0.05		// bottom left
	};

	// Source triangles.
	unsigned short sourceTriangles[] = {
		0, 1, 2,
		0, 2, 3,
		0, 3, 4,
		0, 4, 1
	};

	// Source normals.
	float sourceNormals[] = {
		0.f, 0.f, 1.f,				// source
		-0.4082, 0.4082, 0.8165,	// top left
		0.4082, 0.4082, 0.8165,		// top right
		0.4082, -0.4082, 0.8165,	// bottom right
		-0.4082, -0.4082, 0.8165	// bottom left
	};

	// Create and bind the vertex array object.
	glGenVertexArrays(1, &this->vaoSource);
	glBindVertexArray(this->vaoSource);

	// Create and bind a buffer object for vertex positions.
	glGenBuffers(1, &this->vertexBufferIDSource);
	glBindBuffer(GL_ARRAY_BUFFER, this->vertexBufferIDSource);
	// Copy data into the buffer.
	glBufferData(GL_ARRAY_BUFFER, sizeof(sourceVertices), sourceVertices, GL_STATIC_DRAW);
	// Setup vertex attributes.
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

	// Create and bind a buffer object for vertex indices.
	glGenBuffers(1, &this->faceBufferIDSource);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->faceBufferIDSource);
	// Copy data into the buffer.
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(sourceTriangles), sourceTriangles, GL_STATIC_DRAW);

	// Create and bind a buffer object for vertex normals.
	glGenBuffers(1, &this->normalBufferIDSource);
	glBindBuffer(GL_ARRAY_BUFFER, this->normalBufferIDSource);
	// Copy data into the buffer.
	glBufferData(GL_ARRAY_BUFFER, sizeof(sourceNormals), sourceNormals, GL_STATIC_DRAW);
	// Setup vertex attributes. 
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

	// The vertex array object is set up. Unbind.
	glBindVertexArray(0);

	// Image plane.

	// Image plane corners. Corners are translated by imcenterSrc by the shader
	// before transforming to world coordinates.
	float imagePlaneVertices[] = {
		-this->width / 2.f, this->width / 2.f, 0, // top left
		this->width / 2.f, this->width / 2.f, 0, // top right
		this->width / 2.f, -this->width / 2.f, 0, // bottom right
		-this->width / 2.f, -this->width / 2.f, 0, // bottomleft
	};

	// UV positions of the image plane corners.
	float imagePlaneUV[] = {
		0, 1,
		1, 1,
		1, 0,
		0, 0
	};

	// Image plane triangles.
	unsigned short imagePlaneTriangles[] = {
		0, 3, 2,
		2, 1, 0
	};

	// Create and bind the vertex array object.
	glGenVertexArrays(1, &this->vaoPlane);
	glBindVertexArray(this->vaoPlane);

	// Create and bind a buffer object for vertex positions.
	glGenBuffers(1, &this->vertexBufferIDPlane);
	glBindBuffer(GL_ARRAY_BUFFER, this->vertexBufferIDPlane);
	// Copy data into the buffer.
	glBufferData(GL_ARRAY_BUFFER, sizeof(imagePlaneVertices), imagePlaneVertices, GL_STATIC_DRAW);
	// Setup vertex attributes.
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

	// Create and bind a buffer object for image plane uv positions.
	glGenBuffers(1, &this->uvBufferIDPlane);
	glBindBuffer(GL_ARRAY_BUFFER, this->uvBufferIDPlane);
	// Copy the data into the buffer.
	glBufferData(GL_ARRAY_BUFFER, sizeof(imagePlaneUV), imagePlaneUV, GL_STATIC_DRAW);
	// Setup vertex attributes
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

	// Create and bind a buffer object for vertex indices.
	glGenBuffers(1, &this->faceBufferIDPlane);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->faceBufferIDPlane);
	// Copy data into the buffer.
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(imagePlaneTriangles), imagePlaneTriangles, GL_STATIC_DRAW);

	// The vertex array object is set up. Unbind.
	glBindVertexArray(0);

	// Generate a texture of single value floats.
	this->texture = gl_gen_texture2D(this->nPix);
}

void SourceImage::setup_cuda()
{
	checkCudaErrors(
		cudaGraphicsGLRegisterImage(&this->cudaTexResource, this->texture,
			GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));
	
	checkCudaErrors(
		cudaGraphicsMapResources(1, &this->cudaTexResource));

	checkCudaErrors(
		cudaGraphicsSubResourceGetMappedArray(
			&this->cudaTexArray, this->cudaTexResource, 0, 0));

	this->texBufferSize = this->nPix*this->nPix*sizeof(float);
	checkCudaErrors(cudaMalloc(&this->cudaTexBuffer, this->texBufferSize));

    // Unmap the texture resource so it can be rendered by OpenGL.
    checkCudaErrors(
        cudaGraphicsUnmapResources(1, &this->cudaTexResource));
}

void SourceImage::draw_plane()
{
    checkCudaErrors(
        cudaGraphicsMapResources(1, &this->cudaTexResource));

    checkCudaErrors(
        cudaGraphicsSubResourceGetMappedArray(
            &this->cudaTexArray, this->cudaTexResource, 0, 0));

	// copy the texture buffer to the mapped texture array
	checkCudaErrors(
		cudaMemcpyToArray(
			this->cudaTexArray, 0, 0, this->cudaTexBuffer, this->texBufferSize, 
			cudaMemcpyDeviceToDevice));

    // Unmap the texture resource so it can be rendered by OpenGL.
    checkCudaErrors(
        cudaGraphicsUnmapResources(1, &this->cudaTexResource));

	// draw the image plane
	glBindVertexArray(this->vaoPlane);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, nullptr);
	glBindVertexArray(0);
}

void SourceImage::draw_source()
{
	// draw the x-ray source
	glBindVertexArray(this->vaoSource);
	glDrawElements(GL_LINE_LOOP, 12, GL_UNSIGNED_SHORT, nullptr);
	glBindVertexArray(0);
}
