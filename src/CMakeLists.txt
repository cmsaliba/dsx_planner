file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${dsx_planner_SOURCE_DIR}/include/*.h" "${dsx_planner_SOURCE_DIR}/include/*.hpp")

add_executable(dsx_planner
    dsx_planner.cpp
    DistanceField.h DistanceField.cpp
    glUtils.h glUtils.cpp
    Mesh.h Mesh.cpp
    RenderObject.h RenderObject.cpp
    SourceImage.h SourceImage.cpp
    kernels.cuh kernels.cu
    objVertexShader.glsl objFragmentShader.glsl
    ipVertexShader.glsl ipFragmentShader.glsl

    ${HEADER_LIST}
)

target_include_directories(dsx_planner
    PUBLIC 
    ../include
)

target_link_libraries(dsx_planner 
    PRIVATE
    cuda.lib
    cudart.lib
    GLEW::GLEW
    glfw
    glm::glm
    assimp::assimp
)

set_property(TARGET dsx_planner PROPERTY CUDA_ARCHITECTURES 70 75 86 89)

add_custom_command(TARGET dsx_planner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/ipVertexShader.glsl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ipVertexShader.glsl
    COMMENT "Created ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ipVertexShader.glsl"
)

add_custom_command(TARGET dsx_planner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/ipFragmentShader.glsl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ipFragmentShader.glsl
    COMMENT "Created ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ipFragmentShader.glsl"
)

add_custom_command(TARGET dsx_planner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/objVertexShader.glsl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/objVertexShader.glsl
    COMMENT "Created ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/objVertexShader.glsl"
)

add_custom_command(TARGET dsx_planner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/objFragmentShader.glsl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/objFragmentShader.glsl
    COMMENT "Created ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/objFragmentShader.glsl"
)

add_custom_command(TARGET dsx_planner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/sample ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/
    COMMENT "Copied sample directory."
)

# IDEs should put the headers in a nice place.
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})