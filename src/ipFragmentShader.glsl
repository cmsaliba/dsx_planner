#version 450 core

// Interpolated values from the vertex shaders.
in vec2 UV;

// Output data.
out vec4 outColor;

uniform sampler2D texSampler;

void main()
{
    outColor = vec4(texture(texSampler, UV).rrr, 1.0);
}
