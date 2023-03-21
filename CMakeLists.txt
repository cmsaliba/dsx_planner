cmake_minimum_required(VERSION 3.18)
project(dsx_planner
    LANGUAGES C CXX CUDA
    )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/bin)

# Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
endif()

# ---------- External and third party libraries. -------------------------------

# OpenGL.
find_package(OpenGL REQUIRED)

# GLEW
find_package(GLEW REQUIRED)

# GLFW
find_package(glfw3 CONFIG REQUIRED)

# glm
find_package(glm CONFIG REQUIRED)

# ASSIMP
find_package(assimp CONFIG REQUIRED)

# ---------- DSX-Planner -------------------------------------------------------
add_subdirectory(src)