#version 450 core
layout(location = 0) in vec3 vertexPosition_modelSpace;
layout(location = 1) in vec3 vertexNormal_modelSpace;

out vec3 color;
out vec3 Position_worldSpace;
out vec3 Normal_cameraSpace;
out vec3 EyeDirection_cameraSpace;
out vec3 LightDirection_cameraSpace;


uniform vec3 objColor;
uniform mat4 MVP;
uniform mat4 V;
uniform mat4 M;

void main()
{
	// Output position of the vertex, in clip space : MVP * position
    gl_Position = MVP * vec4(vertexPosition_modelSpace, 1.0);

	// Position of the vertex, in worldspace : M * position
	Position_worldSpace = (M * vec4(vertexPosition_modelSpace,1)).xyz;

	// Vector that goes from the vertex to the camera, in camera space.
	// In camera space, the camera is at the origin (0,0,0).
	vec3 vertexPosition_cameraSpace = ( V * M * vec4(vertexPosition_modelSpace,1)).xyz;
	EyeDirection_cameraSpace = vec3(0,0,0) - vertexPosition_cameraSpace;

	// Vector that goes from the vertex to the light, in camera space. Light is at the camera.
	LightDirection_cameraSpace = EyeDirection_cameraSpace;

	// Normal of the the vertex, in camera space
	Normal_cameraSpace = ( V * M * vec4(vertexNormal_modelSpace,0)).xyz; // Only correct if ModelMatrix does not scale the model ! Use its inverse transpose if not.

	color = objColor;
}
