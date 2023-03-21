# dsx_planner
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[![DOI](https://zenodo.org/badge/617029392.svg)](https://zenodo.org/badge/latestdoi/617029392)



## Introduction
This is a basic planning tool for previewing dynamic stereographic x-ray experiments.

## Requirements
* CUDA capable GPU
* [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone) Toolkit and Driver
* Third-Party Dependencies
    * [GLM](https://glm.g-truc.net/0.9.9/) Library
    * [GLEW](https://glew.sourceforge.net/) Library
    * [GLFW](https://www.glfw.org/) Library
    * [ASSIMP](https://assimp-docs.readthedocs.io/en/v5.1.0/) Library
* C/C++ compiler
    * Windows - [MSVC](https://www.visualstudio.com/)
    * Linux - [GNU compiler](https://gcc.gnu.org/)
* [CMake](https://cmake.org/) cross-platform build system
* (Optional) [VCPKG](https://vcpkg.io/en/index.html) Windows package installer

## Windows Builds
Tested using CUDA 11.8 with Visual Studio 2019.

### Dependencies
1. Install CUDA Toolkit and NVIDIA driver.
2. Use VCPKG to install dependencies (or alternatively build from source).
    * ./vcpkg install glm:x64-windows
    * ./vcpkg install glew:x64-windows
    * ./vcpkg install glfw3:x64-windows 
    * ./vcpkg install assimp:x64-windows

### CMake Build using VS2019 and VCPKG
1. Open the CMake GUI and set the source code location to *path_to_dsx_planner* and the build directory to *path_to_dsx_planner/build*.
2. Press Configure and select VS2019 x64 and Specify toolchain for cross-compiling.
3. Select the *vcpkg/scripts/buildsystems/vcpkg.cmake* file from the VCPKG install location.
4. Press Configure.
5. Press Generate.
6. Open the generated VS2019 solution.
7. Build the dsx_planner project in the desired configuration.

*Note: Newer graphics cards may require updating the desired CUDA architectures in the *src/CmakeLists.txt* file.

## Usage

### Command Line
With the *dsx_planner* executable in your current directory.
```
dsx_planner pointer_file.txt
```

### Pointer File
The pointer file sets up the simulation. The structure of the file is as follows:

* Number of Source/Image Intensifer Pairs (N)
* Number of Objects (M)
* Object 0 Name
* Object 0 Density
* Object 1 Name
* Object 1 Density
* ...
* Object M Name
* Object M Density
* Config Directory
* OBJ Directory
* CSV Directory

The Config Directory contains the configuration files for each of the source/image intensifier pairs. The structure of the configuration files is as follows:

* Pose matrix of the source.
* Position of the image center in source coordinates.
* Size of the image intensifier.
* Image size in pixels.

The OBJ Directory contains the meshes for each of the objects.

The CSV Directory contains the pose sequence files for each of the objects.

### Sample Files
The *sample* directory contains sample files for a walking simulation with the left and right femur and tibia rendered in an environment with four source/image intensifier pairs. The sample files are copied to the *bin* directory after building the project. To run the sample simulation, execute the following command from the *bin* directory:
```
dsx_planner pointer_file.txt
```

### Manipulating the Simulation
**Playback**
* Pause/Resume: Space
* Skip Forward/Backward: Right/Left Arrow

**View**
* Rotate: Left-Click and Drag
* Pan: Right-Click and Drag
* Zoom: Scroll

**Selection**
* Select Source(s): Double Left-Click (multiple can be selected at once)
* Deselect All: Double Left-Click on empty space or Esc

**Manipulate Source/Image Intensifiers**

With one or more sources selected:
* Rotate: Left-Click and Drag
* Translate: Right-Click and Drag
* Change Source-Image Distance: Scroll
* Save Current Configuration: Enter
* Reset to Last Saved Configuration: Backspace

*On exiting the simulation the last saved configuration is written to the configuration files in the Config Directory.*
