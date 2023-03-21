#pragma once

#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

// Initialize OpenGL. Create GLFW window and create its OpenGL context.
GLFWwindow* gl_initglfw(int width, int height, const std::string& name);

void gl_initglew(void);

GLuint gl_load_shaders(std::string vertexFile,
                    std::string fragmentFile);

GLuint gl_gen_texture2D(int);
