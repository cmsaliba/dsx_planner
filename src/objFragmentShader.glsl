#version 450 core

in vec3 color;

in vec3 Position_worldSpace;
in vec3 Normal_cameraSpace;
in vec3 EyeDirection_cameraSpace;
in vec3 LightDirection_cameraSpace;

// Output data.
out vec3 outColor;

// Values that stay constant for the whole mesh.
uniform vec3 LightPosition_worldSpace;

void main()
{
	// Light emission properties
	// You probably want to put them as uniforms
	vec3 LightColor = vec3(0.8,0.8,0.8);
	float LightPower = 0.5f;

	// Material Properties
	vec3 MaterialDiffuseColor;
	MaterialDiffuseColor = color;
	vec3 MaterialAmbientColor = vec3(0.5,0.5,0.5) * MaterialDiffuseColor;
	vec3 MaterialSpecularColor = vec3(0.3,0.3,0.3);

	// Distance to the light
	float distance = length( LightPosition_worldSpace - Position_worldSpace );

	// Normal of the computed fragment, in camera space
	vec3 n = normalize( Normal_cameraSpace );
	// Direction of the light (from the fragment to the light)
	vec3 l = normalize( LightDirection_cameraSpace );
	// Cosine of the angle between the normal and the light direction,
	// clamped above 0
	//  - light is at the vertical of the triangle -> 1
	//  - light is perpendicular to the triangle -> 0
	//  - light is behind the triangle -> 0
	float cosTheta = clamp( dot( n,l ), 0,1 );

	// Eye vector (towards the camera)
	vec3 E = normalize(EyeDirection_cameraSpace);
	// Direction in which the triangle reflects the light
	vec3 R = reflect(-l,n);
	// Cosine of the angle between the Eye vector and the Reflect vector,
	// clamped to 0
	//  - Looking into the reflection -> 1
	//  - Looking elsewhere -> < 1
	float cosAlpha = clamp( dot( E,R ), 0,1 );

	outColor =
		// Ambient : simulates indirect lighting
		MaterialAmbientColor +
		// Diffuse : "color" of the object
		MaterialDiffuseColor * LightColor * LightPower * cosTheta / (distance*distance) +
		// Specular : reflective highlight, like a mirror
		MaterialSpecularColor * LightColor * LightPower * pow(cosAlpha,5) / (distance*distance);
}
