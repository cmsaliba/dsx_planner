#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>

#include "Mesh.h"
#include "DistanceField.h"
#include "RenderObject.h"

RenderObject::~RenderObject()
{
    clear_mesh_data(&this->mesh);
}

void RenderObject::setup_gl()
{
    // create and bind the vertex array object
    glGenVertexArrays(1, &this->vao);
    glBindVertexArray(this->vao);

    // create and bind a buffer object for vertex positions
    glGenBuffers(1, &this->vertexBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, this->vertexBufferID);
    // copy data into the buffer
    glBufferData(GL_ARRAY_BUFFER, (this->mesh.nVertices + this->mesh.nFaces) * 3 * sizeof(float), this->mesh.allPoints, GL_STATIC_DRAW);
    // setup vertex attributes
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // create and bind a buffer object for vertex indices
    glGenBuffers(1, &this->faceBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->faceBufferID);
    // copy data into the buffer
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, this->mesh.nFaces * 3 * sizeof(unsigned int), this->mesh.faces, GL_STATIC_DRAW);

    // create and bind a buffer object for vertex normals
    glGenBuffers(1, &this->normalBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, this->normalBufferID);
    // copy data into the buffer
    glBufferData(GL_ARRAY_BUFFER, this->mesh.nVertices * 3 * sizeof(float), this->mesh.normals, GL_STATIC_DRAW);
    // setup vertex attributes
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // the vertex array object is set up, unbind
    glBindVertexArray(0);
}

void RenderObject::draw()
{
    glBindVertexArray(this->vao);
    glDrawElements(GL_TRIANGLES, this->mesh.nFaces * 3, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}