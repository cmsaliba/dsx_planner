#pragma once

#include <vector>

#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>

#include "Mesh.h"
#include "DistanceField.h"

class RenderObject
{
public:
    RenderObject() :
        nFrames(0),
        Model(1),
        MVP(1),
        color(0.8, 0.8, 0.8),
        density(1.0),
        vao(0),
        vertexBufferID(0),
        faceBufferID(0),
        normalBufferID(0)
        {
        }

        ~RenderObject();

        void setup_gl();
        void draw();

        Mesh mesh;
        glm::mat4 Model;
        glm::mat4 MVP;
        glm::vec3 color;

        float density;

        DistanceField distanceField;

        std::vector<float> pose;
        int nFrames;

    private:

        GLuint vao;
        GLuint vertexBufferID;
        GLuint faceBufferID;
        GLuint normalBufferID;    
};