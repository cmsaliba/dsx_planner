#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

// OpenGL
#include <GL/glew.h>        // GL extensions
#include <GLFW/glfw3.h>     // GL window

// CUDA
#include <cuda_runtime.h>       // runtime library
#include <cuda_gl_interop.h>    // interoperability with OpenGL

// Open Asset Import Library
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// GL Math Library
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

// additional CUDA helper functions (included with SDK examples)
#include "helper_cuda.h"
#include "helper_cuda_gl.h"

#include "glUtils.h"
#include "Mesh.h"
#include "DistanceField.h"
#include "RenderObject.h"
#include "SourceImage.h"

#include "kernels.cuh"

// todo: pass by reference instead of pointers? generate dfield, mesh, etc

// globals to modify for OpenGL rendering
int windowWidth(1200), windowHeight(1200);
glm::mat4 View, Projection;
glm::vec3 eye(1, 0, 0);
glm::vec3 center(0, 0, 0);
glm::vec3 up(0, 1, 0);
float viewportWidth(5), viewportHeight(5);
int shouldPan(GL_FALSE), shouldRotate(GL_FALSE);
double initMouseX, initMouseY;
double initScrollY(0);
auto prevRelease(std::chrono::system_clock::now());

bool paused(false);
bool skipFwd(false);
bool skipBwd(false);

std::vector<SourceImage> sourceImages;
int nSelected(0);

/*
* forward declarations
*/

// mouse position in normalized device coordinates, assumes Z position is the
// near clipping plane, position given as a vec3
// get mouse position from glfw
glm::vec3 mouse_ndc(GLFWwindow* window);
// get mouse position from input
glm::vec3 mouse_ndc(GLFWwindow* window, double xpos, double ypos);
// convert normalized device coordinates to world coordinates
glm::vec3 ndc2world(glm::vec3 ndc, bool isVec);

// GLFW window callbacks
void key_callback(GLFWwindow* window, int key, int scancode, int action,
                  int mods);
void mouse_button_callback(GLFWwindow* window, int button, int action,
                           int mods);
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void window_size_callback(GLFWwindow* window, int width, int height);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

// create a Mesh struct from a 3D model file
void generate_mesh(const std::string& filename, Mesh* mesh);
// create a DistanceField struct from a Mesh struct
void generate_distance_field(DistanceField* distanceField, Mesh* mesh);
// load pose data for a render object
void load_tracking(const std::string& fname, std::vector<float>& trackMat);
// load a source-image pair from a file
void load_source_image(const std::string& fname, SourceImage* cam);
// save a source-image pair to a file
void save_source_image(const std::string& fname, SourceImage* cam);
// separate a full path into the directory and the file
void fileparts(const std::string& str, std::string* directory,
               std::string* file);

/*
 * Main 
 */
int main(int argc, char* argv[])
{
  // check that pointer file is given as command line argument
  if (argc < 2)
  {
    printf("Pointer file required.\n");
    return 1;
  }

  // parse the pointer file
  std::string pointerPath, pointerDir, pointerFile;
  pointerPath = argv[1];
  fileparts(pointerPath, &pointerDir, &pointerFile);

  std::ifstream pointer;
  int nCams, nObjs;
  std::vector<std::string> objName;
  std::vector<float> objDensity;
  std::string configDir, objDir, csvDir;
  pointer.open(pointerPath);
  pointer >> nCams;
  pointer >> nObjs;

  std::string tmpName;
  float tmpDensity;
  for (int i = 0; i < nObjs; i++)
  {
    pointer >> tmpName;
    objName.push_back(tmpName);
    pointer >> tmpDensity;
    objDensity.push_back(tmpDensity);
  }

  pointer >> configDir;
  pointer >> objDir;
  pointer >> csvDir;

  // weighting and step distance for the ray tracing (all units in m)
  float weight(0.01f), step(0.001f);

  // initialize GLFW window and GLEW
  GLFWwindow* window = gl_initglfw(windowWidth, windowHeight, "Sim Render");
  glfwGetWindowSize(window, &windowWidth, &windowHeight);
  viewportWidth = ((float)windowWidth / (float)windowHeight) * viewportHeight;
  gl_initglew();

  // setup CUDA for OpenGL interop
  cudaGLSetGLDevice(0);

  // set callbacks for interacting with window
  glfwSetKeyCallback(window, key_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetCursorPosCallback(window, cursor_position_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetWindowSizeCallback(window, window_size_callback);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // shader program for objects in rendering
  GLuint objShaderProgramID = gl_load_shaders("objVertexShader.glsl",
                                              "objFragmentShader.glsl");
  // location of the shader program uniforms
  GLuint objColorID = glGetUniformLocation(objShaderProgramID, "objColor");
  GLuint objMVPID = glGetUniformLocation(objShaderProgramID, "MVP");
  GLuint objVID = glGetUniformLocation(objShaderProgramID, "V");
  GLuint objMID = glGetUniformLocation(objShaderProgramID, "M");
  GLuint objLightPosID = glGetUniformLocation(objShaderProgramID,
                                              "LightPosition_worldSpace");

  // shader program for image planes in rendering
  GLuint ipShaderProgramID = gl_load_shaders("ipVertexShader.glsl",
                                             "ipFragmentShader.glsl");
  // location of the shader program uniforms
  GLuint ipMVPID = glGetUniformLocation(ipShaderProgramID, "MVP");
  GLuint ipTextureID = glGetUniformLocation(ipShaderProgramID, "texSampler");

  // load the render objects (i.e.: bones and other objects in volume)
  std::vector<RenderObject> renderObjects(objName.size());
  int nFrames = std::numeric_limits<int>::max();

  for (int i = 0; i < objName.size(); i++)
  {
    generate_mesh(objDir + objName.at(i) + ".obj", &renderObjects[i].mesh);
    generate_distance_field(&renderObjects[i].distanceField,
                            &renderObjects[i].mesh);
    load_tracking(csvDir + objName.at(i) + ".csv", renderObjects[i].pose);
    renderObjects[i].density = objDensity[i];
    renderObjects[i].nFrames = (int)renderObjects[i].pose.size() / 16;
    renderObjects[i].setup_gl();
    nFrames = std::min(renderObjects[i].nFrames, nFrames);
  }

  // load the source and image plane configurations
  sourceImages.resize(nCams);
  int i = 0;
  for (std::vector<SourceImage>::iterator cam = sourceImages.begin();
       cam != sourceImages.end(); ++cam)
  {
    std::string camFile = configDir + "cam" + std::to_string(i + 1) + ".txt";
    load_source_image(camFile, std::addressof(*cam));
    cam->setup_gl();
    cam->setup_cuda();
    i++;
  }

  // pre-allocate some GPU memory
  float* d_imv, * d_aabb;
  checkCudaErrors(cudaMalloc(&d_imv, 16 * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_aabb, 6 * sizeof(float)));

  // render the simulation
  glfwShowWindow(window);
  int f = 0; // frame
  do
  {
    // frame to display
    f = f % (nFrames * 16);

    // let OpenGL finish
    glFinish();
    glFlush();

    // update the orthographic projection and view matrices
    Projection = glm::ortho(-viewportWidth / 2, viewportWidth / 2,
                            -viewportHeight / 2, viewportHeight / 2,
                            -5.0f, 5.0f);
    View = glm::lookAt(eye, center, up);

    // pose matrices for the render objects
    for (std::vector<RenderObject>::iterator obj = renderObjects.begin();
         obj != renderObjects.end(); ++obj)
    {
      obj->Model = glm::mat4(
                             obj->pose[f + 0], obj->pose[f + 1], obj->pose[f + 2], obj->pose[f + 3],
                             obj->pose[f + 4], obj->pose[f + 5], obj->pose[f + 6], obj->pose[f + 7],
                             obj->pose[f + 8], obj->pose[f + 9], obj->pose[f + 10], obj->pose[f + 11],
                             obj->pose[f + 12], obj->pose[f + 13], obj->pose[f + 14], obj->pose[f + 15]);

      obj->MVP = Projection * View * obj->Model;
    }


    // DRR for each source-image pair
    for (std::vector<SourceImage>::iterator cam = sourceImages.begin();
         cam != sourceImages.end(); ++cam)
    {
      // set up kernel blocks and grid
      dim3 dimBlock(16, 16);
      dim3 dimGrid;
      dimGrid.x = (cam->nPix + dimBlock.x - 1) / dimBlock.x;
      dimGrid.y = (cam->nPix + dimBlock.y - 1) / dimBlock.y;

      // clear the DRR
      clearDRR(dimGrid, dimBlock, cam->cudaTexBuffer, cam->nPix);

      // todo: for real-time update the cam->src2world matrix and
      // cam->imcenterSrc vectors based on the source and image
      // intensifier locations. note: currently the raycast kernel assumes
      // that the normal vector of the intensifier plane is parallel to
      // z-axis of the source

      // raycast all render objects
      for (std::vector<RenderObject>::iterator obj = renderObjects.begin();
           obj != renderObjects.end(); ++obj)
      {
        glm::mat4 src2bone = (glm::inverse(obj->Model)) * cam->src2world;
        checkCudaErrors(cudaMemcpy(d_imv, &src2bone[0][0], 16*sizeof(float),
          cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_aabb, obj->mesh.aabb, 6*sizeof(float),
          cudaMemcpyHostToDevice));

        raycast(dimGrid, dimBlock, obj->distanceField.dField,
                obj->distanceField.offset, obj->distanceField.voxelSize,
                obj->distanceField.nPts, d_imv, &cam->imcenterSrc[0], d_aabb,
                cam->cudaTexBuffer, cam->pixSize, cam->nPix, weight, step,
                obj->density);
      }
    }

    // OpenGL clears
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // draw objects in the scene using the shader program
    glUseProgram(objShaderProgramID);
    glUniformMatrix4fv(objVID, 1, GL_FALSE, &View[0][0]);
    glUniform3f(objLightPosID, eye.x, eye.y, eye.z);

    for (std::vector<RenderObject>::iterator obj = renderObjects.begin();
         obj != renderObjects.end(); ++obj)
    {
      glUniform3f(objColorID, obj->color.x, obj->color.y, obj->color.z);
      glUniformMatrix4fv(objMVPID, 1, GL_FALSE, &obj->MVP[0][0]);
      glUniformMatrix4fv(objMID, 1, GL_FALSE, &obj->Model[0][0]);
      obj->draw();
    }

    // draw the sources and image planes
    for (std::vector<SourceImage>::iterator cam = sourceImages.begin();
         cam != sourceImages.end(); ++ cam)
    {
      // use the render object shader program to draw the sources
      glUseProgram(objShaderProgramID);

      glUniform3f(objColorID, cam->color.x, cam->color.y, cam->color.z);
      cam->sourceMVP = Projection * View * cam->src2world;
      glUniformMatrix4fv(objMVPID, 1, GL_FALSE, &cam->sourceMVP[0][0]);
      glUniformMatrix4fv(objMID, 1, GL_FALSE, &cam->src2world[0][0]);
      cam->draw_source();

      // use the image plane shader program to draw the DRRs as textures
      glUseProgram(ipShaderProgramID);

      // bind the texture in Texture Unit 0
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, cam->texture);

      // set sampler to use Texture Unit 0
      glUniform1i(ipTextureID, 0);

      cam->planeMVP = Projection * View * glm::translate(cam->src2world,
                                                         cam->imcenterSrc);
      glUniformMatrix4fv(ipMVPID, 1, GL_FALSE, &cam->planeMVP[0][0]);
      cam->draw_plane();
    }

    // keep running
    glfwSwapBuffers(window);
    glfwPollEvents();

    // advance frame if not paused
    if (!paused)
      f += 16;
    // if paused, check if should skip forward or backward
    else if (skipFwd && f < 16 * (nFrames - 1))
      f += 16;
    else if (skipBwd && f > 0)
      f -= 16;
  } while (!glfwWindowShouldClose(window));
  glfwDestroyWindow(window);
  glfwTerminate();

  // free pre-allocated CUDA memory
  cudaFree(d_imv);
  cudaFree(d_aabb);

  // save the source and image plane configurations.
  for (int i = 0; i < nCams; i++)
  {
    std::string camFile = configDir + "cam" + std::to_string(i + 1) + ".txt";
    save_source_image(camFile, &sourceImages[i]);
  }

  return 0;
}

// mouse position in normalized device coordinates, assumes Z position is the
// near clipping plane, position given as a vec3
// get mouse position from glfw
glm::vec3 mouse_ndc(GLFWwindow* window)
{
  // get x and y pixel coordinates of the mouse
  // origin top left
  // right - positive x
  // down - positive y
  double xpos, ypos;
  glfwGetCursorPos(window, &xpos, &ypos);

  // convert to normalized device coordinates:
  // origin in center of cube
  // positive x - right
  // positive y - up
  // positive z - into screen
  // *not right-handed coord sys
  glm::vec4 mouseNDC((xpos - windowWidth / 2.f) / (windowWidth / 2.f), // transform to values between -1 and +1
                     -(ypos - windowHeight / 2.f) / (windowHeight / 2.f), // transform to values between -1 and +1, also invert
                     -1, // z is into screen, -1 near clipping plane, +1 far clipping plane
                     1); // w is 1 for a point in homogeneous coords

  return glm::vec3(mouseNDC);
}

// get mouse position from input
glm::vec3 mouse_ndc(GLFWwindow* window, double xpos, double ypos)
{
  // x and y pixel coordinates of the mouse
  // origin top left
  // right - positive x
  // down - positive y

  // convert to normalized device coordinates:
  // origin in center of cube
  // positive x - right
  // positive y - up
  // positive z - into screen
  // *not right-handed coord sys
  glm::vec4 mouseNDC((xpos - windowWidth / 2.f) / (windowWidth / 2.f), // transform to values between -1 and +1
                     -(ypos - windowHeight / 2.f) / (windowHeight / 2.f), // transform to values between -1 and +1, also invert
                     -1, // z is into screen, -1 near clipping plane, +1 far clipping plane
                     1); // w is 1 for a point in homogeneous coords

  return glm::vec3(mouseNDC);
}

// convert normalized device coordinates to world coordinates
glm::vec3 ndc2world(glm::vec3 ndc, bool isVec)
{
  // multiply by the inverse of the view-projection matrix to
  // transform to world coordinates
  glm::vec4 world;
  if (isVec) // converting a vector, w=0 in homogeneous representation
  {
    world = glm::inverse(Projection * View) * glm::vec4(ndc, 0.f);
  }
  else // converting a point, w=1 in homogeneous representation
  {
    world = glm::inverse(Projection * View) * glm::vec4(ndc, 1.f);

    // perform persepective division (ie undo scaling if using 
    // perspective projection)
    world /= world.w;
  }

  return glm::vec3(world);
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
  viewportWidth = ((float)width / (float)height) * viewportHeight;

  windowWidth = width;
  windowHeight = height;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
  glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action,
                  int mods)
{
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
    paused = !paused;
  else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
  {
    skipFwd = true;
    skipBwd = false;
  }
  else if (key == GLFW_KEY_RIGHT && action == GLFW_RELEASE)
    skipFwd = false;
  else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
  {
    skipBwd = true;
    skipFwd = false;
  }
  else if (key == GLFW_KEY_LEFT && action == GLFW_RELEASE)
    skipBwd = false;
  else if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
  {
    nSelected = 0;
    for (int i = 0; i < sourceImages.size(); i++)
    {
      sourceImages[i].selected = false;
      sourceImages[i].color = glm::vec3(0.f, 1.f, 0.f);
    }
  }
  else if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
  {
    nSelected = 0;
    for (int i = 0; i < sourceImages.size(); i++)
    {
      sourceImages[i].selected = false;
      sourceImages[i].color = glm::vec3(0.f, 1.f, 0.f);
      sourceImages[i].src2world_saved = sourceImages[i].src2world;
      sourceImages[i].imcenterSrc_saved = sourceImages[i].imcenterSrc;
    }
  }
  else if (key == GLFW_KEY_BACKSPACE && action == GLFW_PRESS)
  {
    // reset with most recent saved configurations
    eye = glm::vec3(1, 0, 0);
    center = glm::vec3(0, 0, 0);
    up = glm::vec3(0, 0, 1);

    nSelected = 0;
    for (int i = 0; i < sourceImages.size(); i++)
    {
      sourceImages[i].selected = false;
      sourceImages[i].color = glm::vec3(0.f, 1.f, 0.f);
      sourceImages[i].src2world = sourceImages[i].src2world_saved;
      sourceImages[i].imcenterSrc = sourceImages[i].imcenterSrc_saved;
    }
  }
  else if (key == GLFW_KEY_1 && action == GLFW_PRESS)
    eye = glm::vec3(1.f, 0.f, 0.f) + center;
  else if (key == GLFW_KEY_2 && action == GLFW_PRESS)
    eye = glm::vec3(-1.f, 0.f, 0.f) + center;
  else if (key == GLFW_KEY_3 && action == GLFW_PRESS)
    eye = glm::vec3(0.f, 1.f, 0.f) + center;
  else if (key == GLFW_KEY_4 && action == GLFW_PRESS)
    eye = glm::vec3(0.f, -1.f, 0.f) + center;
  else if (key == GLFW_KEY_5 && action == GLFW_PRESS)
    eye = glm::vec3(0.f, 0.f, 1.f) + center;
  else if (key == GLFW_KEY_6 && action == GLFW_PRESS)
    eye = glm::vec3(0.f, 0.f, -1.f) + center;
  else if (key == GLFW_KEY_7 && action == GLFW_PRESS)
    up = glm::vec3(1.f, 0.f, 0.f);
  else if (key == GLFW_KEY_8 && action == GLFW_PRESS)
    up = glm::vec3(0.f, 1.f, 0.f);
  else if (key == GLFW_KEY_9 && action == GLFW_PRESS)
    up = glm::vec3(0.f, 0.f, 1.f);
  else if (key == GLFW_KEY_0 && action == GLFW_PRESS && nSelected == 1)
  {
    for (int i = 0; i < sourceImages.size(); i++)
    {
      if (sourceImages[i].selected)
      {
        eye = glm::vec3(sourceImages[i].src2world[3]);
        center = eye - glm::vec3(sourceImages[i].src2world[2]);
        up = glm::vec3(sourceImages[i].src2world[1]);
      }
    }
  }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
  float diff = yoffset - initScrollY;

  if (nSelected == 0)
  {
    viewportWidth += diff * 0.1 * viewportWidth;
    viewportHeight += diff * 0.1 * viewportHeight;
  }
  else
  {
    for (int i = 0; i < sourceImages.size(); i++)
    {
      if (sourceImages[i].selected)
      {
        sourceImages[i].imcenterSrc[2] -= 0.01 * diff;
      }
    }
  }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
  {
    auto now = std::chrono::system_clock::now();
    double diff_ms =
      std::chrono::duration<double, std::milli>(now - prevRelease).count();
    prevRelease = now;
    if (diff_ms > 10 && diff_ms < 200)
    {
      // cursor position in normalized device coordinates
      glm::vec3 mouseNDC = mouse_ndc(window);

      // cursor position in world coordinates
      glm::vec3 origin = ndc2world(mouseNDC, false); // false bc point not vector

      // get the direction of a vector pointing into the screen in world
      // space
      glm::vec3 ray = glm::normalize(
                                     ndc2world(glm::vec3(0, 0, 1), // vector into screen in normalized device coordinates (+z into screen)
                                               true) // vector not a point
                                    );

      int nCollis = 0;
      // objects to select are the xray sources (line sphere collision)
      for (int i = 0; i < sourceImages.size(); i++)
      {
        glm::vec3 srcCenter = glm::vec3(
                                        sourceImages[i].src2world[3][0],
                                        sourceImages[i].src2world[3][1],
                                        sourceImages[i].src2world[3][2]);

        float rad = sourceImages[i].selectRad;

        float determinate =
          std::pow(glm::dot(ray, origin - srcCenter), 2)
          - std::pow(glm::length(origin - srcCenter), 2)
          + std::pow(rad, 2);

        if (determinate > 0) // collision
        {
          nCollis += 1;
          sourceImages[i].selected = !sourceImages[i].selected;
          if (sourceImages[i].selected)
          {
            nSelected += 1;
            sourceImages[i].color = glm::vec3(1.f, 0.f, 0.f);
          }
          else
          {
            nSelected -= 1;
            sourceImages[i].color = glm::vec3(0.f, 1.f, 0.f);
          }
        }
      }

      // if double click on empty, clear selected cameras and select world
      if (nCollis == 0)
      {
        nSelected = 0;
        for (int i = 0; i < sourceImages.size(); i++)
        {
          sourceImages[i].selected = false;
          sourceImages[i].color = glm::vec3(0.f, 1.f, 0.f);
        }
      }
    }
    shouldPan = GL_FALSE;
    shouldRotate = GL_FALSE;
  }
  else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
  {
    shouldPan = GL_TRUE;
    shouldRotate = GL_FALSE;
    glfwGetCursorPos(window, &initMouseX, &initMouseY);
  }
  else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
  {
    shouldPan = GL_FALSE;
    shouldRotate = GL_TRUE;
    glfwGetCursorPos(window, &initMouseX, &initMouseY);
  }
  else
  {
    shouldPan = GL_FALSE;
    shouldRotate = GL_FALSE;
  }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
  if (nSelected == 0)
  {
    glm::mat4 invView = glm::inverse(View);
    glm::vec3 cameraX = glm::vec3(invView[0][0], invView[0][1], invView[0][2]);
    glm::vec3 cameraY = glm::vec3(invView[1][0], invView[1][1], invView[1][2]);

    up = cameraY;

    if (shouldPan)
    {
      // translate camera and scene center opposite to the cursor
      glfwGetWindowSize(window, &windowWidth, &windowHeight);
      float xTrans = viewportWidth * (xpos - initMouseX) / windowWidth;
      float yTrans = viewportHeight * (ypos - initMouseY) / windowHeight;
      eye = eye - xTrans * cameraX + yTrans * cameraY;
      center = center - xTrans * cameraX + yTrans * cameraY;
      // update cursor position
      glfwGetCursorPos(window, &initMouseX, &initMouseY);
    }
    else if (shouldRotate)
    {
      glfwGetWindowSize(window, &windowWidth, &windowHeight);

      glm::vec3 P1 = glm::vec3(
                               (initMouseX / windowWidth) * 2.0 - 1.0,
                               -((initMouseY / windowHeight) * 2.0 - 1.0),
                               0
                              );

      glm::vec3 P2 = glm::vec3(
                               (xpos / windowWidth) * 2.0 - 1.0,
                               -((ypos / windowHeight) * 2.0 - 1.0),
                               0
                              );

      glm::vec4 transVec(0);

      transVec.x = sin(5.f * (P2.x - P1.x));
      transVec.y = sin(5.f * (P2.y - P1.y));
      transVec.z = 1.f - cos(5.f * (P2.x - P1.x)) + 1.f
        - cos(5.f * (P2.y - P1.y));

      transVec = invView * transVec;

      eye = eye - glm::vec3(transVec.x, transVec.y, transVec.z);

      // update cursor position
      glfwGetCursorPos(window, &initMouseX, &initMouseY);
    }
  }
  else
  {
    if (shouldPan)
    {
      // intial and final positions of the cursor in world space
      glm::vec3 initPos = ndc2world(
                                    mouse_ndc(window, initMouseX, initMouseY),
                                    false);

      glm::vec3 newPos = ndc2world(
                                   mouse_ndc(window, xpos, ypos),
                                   false);

      // cursor translation in world space
      glm::vec3 trans = newPos - initPos;


      // if the source is selected, translate with the cursor by updating
      // the pose matrix so the image plane moves with the source
      for (int i = 0; i < sourceImages.size(); i++)
      {
        if (sourceImages[i].selected)
        {
          sourceImages[i].src2world =
            glm::translate(glm::mat4(1.f), trans) * sourceImages[i].src2world;
        }
      }

      // update cursor position
      glfwGetCursorPos(window, &initMouseX, &initMouseY);
    }
    else if (shouldRotate)
    {
      // cursor translation in world space
      glm::vec3 initPos = ndc2world(
                                    mouse_ndc(window, initMouseX, initMouseY),
                                    false);

      glm::vec3 newPos = ndc2world(
                                   mouse_ndc(window, xpos, ypos),
                                   false);


      for (int i = 0; i < sourceImages.size(); i++)
      {
        if (sourceImages[i].selected)
        {
          // project initial cursor position onto source axes
          glm::vec3 initPosSrc = glm::vec3(
                                           glm::inverse(sourceImages[i].src2world)
                                           * glm::vec4(initPos, 1.f)
                                          );
          // flip rotation direction is cursor is in front of the source
          float flip = initPosSrc.z / abs(initPosSrc.z);

          // project cursor movement onto source axes
          glm::vec3 trans = glm::vec3(
                                      glm::inverse(sourceImages[i].src2world)
                                      * glm::vec4(newPos - initPos, 0.f)
                                     );

          // x translation controls yaw
          sourceImages[i].src2world = glm::rotate(sourceImages[i].src2world,
                                                  flip * trans.x, glm::vec3(0.f, 1.f, 0.f));
          // y translation controls pitch
          sourceImages[i].src2world = glm::rotate(sourceImages[i].src2world,
                                                  -flip * trans.y, glm::vec3(1.f, 0.f, 0.f));
        }
      }
      // update cursor position
      glfwGetCursorPos(window, &initMouseX, &initMouseY);
    }
  }
}

// create a Mesh struct from a 3D model file
void generate_mesh(const std::string& filename, Mesh* mesh)
{
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(filename, aiProcess_Triangulate |
                                           aiProcess_JoinIdenticalVertices | aiProcess_GenSmoothNormals);

  if (!scene || scene->mFlags == AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
  {
    std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
    exit(1);
  }

  const aiMesh* m = scene->mMeshes[0];

  mesh->nVertices = m->mNumVertices;
  mesh->nFaces = m->mNumFaces;

  mesh->allPoints = new float[mesh->nVertices * 3 + mesh->nFaces * 3];

  mesh->vertices = &mesh->allPoints[0];
  mesh->centroids = &mesh->allPoints[mesh->nVertices * 3];
  mesh->normals = new float[mesh->nVertices * 3];
  mesh->faces = new unsigned int[mesh->nFaces * 3];
  mesh->faceAreas = new float[mesh->nFaces];

  float maxX = m->mVertices[0].x;
  float maxY = m->mVertices[0].y;
  float maxZ = m->mVertices[0].z;

  float minX = maxX;
  float minY = maxY;
  float minZ = maxZ;

  float x, y, z;


  for (int i = 0; i < mesh->nVertices; i++)
  {
    x = m->mVertices[i].x;
    y = m->mVertices[i].y;
    z = m->mVertices[i].z;

    minX = (x < minX) ? x : minX;
    minY = (y < minY) ? y : minY;
    minZ = (z < minZ) ? z : minZ;

    maxX = (x > maxX) ? x : maxX;
    maxY = (y > maxY) ? y : maxY;
    maxZ = (z > maxZ) ? z : maxZ;


    mesh->vertices[3 * i] = x;
    mesh->vertices[3 * i + 1] = y;
    mesh->vertices[3 * i + 2] = z;
  }

  mesh->aabb[0] = minX;
  mesh->aabb[1] = minY;
  mesh->aabb[2] = minZ;

  mesh->aabb[3] = maxX;
  mesh->aabb[4] = maxY;
  mesh->aabb[5] = maxZ;

  for (int i = 0; i < mesh->nVertices; i++)
  {
    x = m->mNormals[i].x;
    y = m->mNormals[i].y;
    z = m->mNormals[i].z;

    mesh->normals[3 * i] = x;
    mesh->normals[3 * i + 1] = y;
    mesh->normals[3 * i + 2] = z;
  }

  for (int i = 0; i < mesh->nFaces; i++)
  {
    float x0, x1, x2, y0, y1, y2, z0, z1, z2;

    aiFace face = m->mFaces[i];
    mesh->faces[3 * i] = face.mIndices[0];
    mesh->faces[3 * i + 1] = face.mIndices[1];
    mesh->faces[3 * i + 2] = face.mIndices[2];

    x0 = m->mVertices[face.mIndices[0]].x;
    y0 = m->mVertices[face.mIndices[0]].y;
    z0 = m->mVertices[face.mIndices[0]].z;

    x1 = m->mVertices[face.mIndices[1]].x;
    y1 = m->mVertices[face.mIndices[1]].y;
    z1 = m->mVertices[face.mIndices[1]].z;

    x2 = m->mVertices[face.mIndices[2]].x;
    y2 = m->mVertices[face.mIndices[2]].y;
    z2 = m->mVertices[face.mIndices[2]].z;

    mesh->centroids[3 * i] = (x0 + x1 + x2) / 3.f;
    mesh->centroids[3 * i + 1] = (y0 + y1 + y2) / 3.f;
    mesh->centroids[3 * i + 2] = (z0 + z1 + z2) / 3.f;
  }
}

// create a DistanceField struct from a Mesh struct
void generate_distance_field(DistanceField* distanceField, Mesh* mesh)
{
  // distance field bounding box scale factor
  float scale = 0.1f;

  // size of the axis aligned bounding box of the mesh
  float bbox[3] = {
    mesh->aabb[3] - mesh->aabb[0],
    mesh->aabb[4] - mesh->aabb[1],
    mesh->aabb[5] - mesh->aabb[2]
  };

  // start and end points of the distance field bounding box
  float startPt[3] = {
    mesh->aabb[0] - bbox[0] * scale,
    mesh->aabb[1] - bbox[1] * scale,
    mesh->aabb[2] - bbox[2] * scale
  };
  float endPt[3] = {
    mesh->aabb[3] + bbox[0] * scale,
    mesh->aabb[4] + bbox[1] * scale,
    mesh->aabb[5] + bbox[2] * scale
  };

  // offset of the distance field
  distanceField->offset[0] = startPt[0];
  distanceField->offset[1] = startPt[1];
  distanceField->offset[2] = startPt[2];

  // size of the distance field
  distanceField->dFieldSize[0] = endPt[0] - startPt[0];
  distanceField->dFieldSize[1] = endPt[1] - startPt[1];
  distanceField->dFieldSize[2] = endPt[2] - startPt[2];

  // desired voxel size of the distance field
  float desVoxSize = 0.001; // 1mm

  // number of points in each dimension of the distance field
  distanceField->nPts[0] = ceil(distanceField->dFieldSize[0] / desVoxSize)
    + 1;
  distanceField->nPts[1] = ceil(distanceField->dFieldSize[1] / desVoxSize)
    + 1;
  distanceField->nPts[2] = ceil(distanceField->dFieldSize[2] / desVoxSize)
    + 1;

  // actual voxel size
  distanceField->voxelSize[0] =
    distanceField->dFieldSize[0] / (float)distanceField->nPts[0];
  distanceField->voxelSize[1] =
    distanceField->dFieldSize[1] / (float)distanceField->nPts[1];
  distanceField->voxelSize[2] =
    distanceField->dFieldSize[2] / (float)distanceField->nPts[2];

  // allocate memory on the GPU for the distance field
  distanceField->dFieldBufferSize = (distanceField->nPts[0]
    * distanceField->nPts[1] * distanceField->nPts[2]) * sizeof(float);
  checkCudaErrors(
    cudaMalloc(&distanceField->dField, distanceField->dFieldBufferSize));

  // allocate memory on the GPU for the vertices & centroids, and faces of the
  // mesh
  float* meshPointBuffer;
  checkCudaErrors(cudaMalloc(&meshPointBuffer, (mesh->nVertices + mesh->nFaces)*3*sizeof(float)));
  unsigned int* meshFaceBuffer;
  checkCudaErrors(cudaMalloc(&meshFaceBuffer, mesh->nFaces*3*sizeof(unsigned int)));

  // copy points and faces from the mesh to the GPU
  checkCudaErrors(cudaMemcpy(meshPointBuffer, mesh->allPoints, (mesh->nVertices + mesh->nFaces)*3*sizeof(float), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(meshFaceBuffer, mesh->faces, mesh->nFaces*3*sizeof(unsigned int), cudaMemcpyHostToDevice));

  dim3 dimBlock(10, 10, 10);
  dim3 dimGrid;
  dimGrid.x = (distanceField->nPts[0] + dimBlock.x - 1) / dimBlock.x;
  dimGrid.y = (distanceField->nPts[1] + dimBlock.y - 1) / dimBlock.y;
  dimGrid.z = (distanceField->nPts[2] + dimBlock.z - 1) / dimBlock.z;

  int step, nIter;

  std::cout << "Generating distance field..." << std::flush;
  step = 1000;
  nIter = (mesh->nVertices + mesh->nFaces) / step + 1;
  for (int i = 0; i < nIter; i++)
  {
    calculate_unsigned_distances(dimGrid, dimBlock,
                                 distanceField->dField,
                                 distanceField->offset,
                                 distanceField->voxelSize,
                                 distanceField->nPts, meshPointBuffer,
                                 mesh->nVertices + mesh->nFaces,
                                 i * step, 1000);

    checkCudaErrors(cudaDeviceSynchronize());
  }
  std::cout << "done." << std::endl;

  std::cout << "Signing distance field..." << std::flush;
  step = 100;
  nIter = mesh->nFaces / step + 1;
  for (int i = 0; i < nIter; i++)
  {
    sign_distances(dimGrid, dimBlock, distanceField->dField,
                   distanceField->offset, distanceField->voxelSize,
                   distanceField->nPts, meshPointBuffer, meshFaceBuffer,
                   mesh->nFaces, i * step, step);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  std::cout << "done." << std::endl;

  // free the points and faces allocated on the GPU
  cudaFree(meshPointBuffer);
  cudaFree(meshFaceBuffer);
}

// load bone tracking data.
void load_tracking(const std::string& fname, std::vector<float>& trackMat)
{
	std::ifstream file(fname.c_str());
	if (file.is_open() == false)
	{
		std::cerr << "File not found: " << fname << std::endl;
	}
	std::string csvLine, csvVal;
	float val;
	for (size_t i = 0; std::getline(file, csvLine); ++i)
	{
		std::istringstream csvLineStream(csvLine);
		for (size_t j = 0; std::getline(csvLineStream, csvVal, ','); ++j)
		{
			std::istringstream csvValStream(csvVal);
			csvValStream >> val;
			trackMat.push_back(val);
		}
	}
}

// load a source-image pair from a file
void load_source_image(const std::string& fname, SourceImage* cam)
{
  std::ifstream file(fname.c_str());
  if (file.is_open() == false)
  {
    std::cerr << "File not found: " << fname << std::endl;
  }

  std::string camMatLine, camMatVal, posVecLine, posVecVal;
  float val;
  float camMat[16];
  float posVec[3];

  std::getline(file, camMatLine);
  std::istringstream camMatLineStream(camMatLine);
  for (int i = 0; std::getline(camMatLineStream, camMatVal, ',') && i < 16;
       ++i)
  {
    std::istringstream camMatValStream(camMatVal);
    camMatValStream >> val;
    camMat[i] = val;
  }

  cam->src2world = glm::make_mat4(camMat);
  cam->src2world_saved = glm::make_mat4(camMat);

  std::getline(file, posVecLine);
  std::istringstream posVecLineStream(posVecLine);
  for (int i = 0; std::getline(posVecLineStream, posVecVal, ',') && i < 3;
       ++i)
  {
    std::istringstream posVecValStream(posVecVal);
    posVecValStream >> val;
    posVec[i] = val;
  }

  cam->imcenterSrc = glm::make_vec3(posVec);
  cam->imcenterSrc_saved = glm::make_vec3(posVec);

  file >> cam->width;
  file >> cam->nPix;
  cam->pixSize = cam->width / cam->nPix;
}

// save a source-image pair to a file
void save_source_image(const std::string& fname, SourceImage* cam)
{
  std::ofstream file(fname.c_str());
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      file << cam->src2world_saved[i][j] << ",";
    }
  }
  file << std::endl;
  for (int i = 0; i < 3; i++)
  {
    file << cam->imcenterSrc_saved[i] << ",";
  }
  file << std::endl;
  file << cam->width << std::endl;
  file << cam->nPix << std::endl;
  file.close();
}

// separate a full path into the directory and the file
void fileparts(const std::string& str, std::string* directory, std::string* file)
{
  int found;
  found = str.find_last_of("/\\");
  *directory = str.substr(0, found) + "/";
  *file = str.substr(found + 1);
}
