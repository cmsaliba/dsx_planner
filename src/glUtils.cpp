#include <iostream>
#include <fstream>
#include <vector>
#include <string>

//#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "glUtils.h"

// Initialize OpenGL. Create GLFW window and create its OpenGL context.
GLFWwindow* gl_initglfw(int width, int height, const std::string& name)
{
	// Initialize GLFW.
	if (!glfwInit())
	{
		std::cerr << "Failed to initialize GLFW." << std::endl;
		std::cin.get(); // pause the console screen
	}

	// GLFW window hints.
	glfwWindowHint(GLFW_SAMPLES, 4); // 4x anti-aliasing
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // OpenGL version 4.5
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // context only supports new core functionality
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE); // window dimensions are unlocked
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

	// Open a GLFW window and create its OpenGL context.
	GLFWwindow* window;
	window = glfwCreateWindow(width, height, name.c_str(), nullptr, nullptr);
	if (window == nullptr)
	{
		std::cerr << "Failed to open GLFW window. Using OpenGL version 4.5. Check that your hardware is compatible." << std::endl;
		std::cin.get(); // pause the console screen
		glfwTerminate();
	}
	glfwMakeContextCurrent(window);

	return window;
}

void gl_initglew(void)
{
	// Initialize GLEW after the GLFW window (the OpenGL context) has been created.
	glewExperimental = GL_TRUE; // needed for core profile (to enable modern OpenGL functionality)
	if (glewInit() != GLEW_OK)
	{
		std::cerr << "Failed to initialize GLEW." << std::endl;
		std::cin.get(); // pause the console screen
	}

	// Tell GL to only draw onto a pixel if the shape is closer to the viewer
	glEnable(GL_DEPTH_TEST); // enable depth-testing
	glDepthFunc(GL_LESS); // depth-testing interprets a smaller value as "closer"

	// default to blue background
	glClearColor(0.65f, 0.8f, 1.0f, 1.0f);
}

GLuint gl_load_shaders(std::string vertexFile,
                    std::string fragmentFile)
{
	GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

	// Read vertex shader source code.
	std::string vertexShaderCode;
	std::ifstream vertexShaderStream(vertexFile.c_str());
	if (!vertexShaderStream.is_open())
	{
		std::cout << "Cannot open file: " << vertexFile << std::endl;
		exit(1);
	}
	vertexShaderCode = std::string(
		std::istreambuf_iterator<char>(vertexShaderStream),
		std::istreambuf_iterator<char>());
	vertexShaderStream.close();

	// Read fragment shader source code.
	std::string fragmentShaderCode;
	std::ifstream fragmentShaderStream(fragmentFile.c_str());
	if (!fragmentShaderStream.is_open())
	{
		std::cout << "Cannot open file: " << fragmentFile << std::endl;
		exit(1);
	}
	fragmentShaderCode = std::string(
		std::istreambuf_iterator<char>(fragmentShaderStream),
		std::istreambuf_iterator<char>());
	fragmentShaderStream.close();

	// Compile the vertex shader.
	const GLchar* vertexSource = (const GLchar*)vertexShaderCode.c_str();
	glShaderSource(vertexShaderID, 1, &vertexSource, nullptr);
	glCompileShader(vertexShaderID);

	// Check the vertex shader.
	GLint isCompiled(GL_FALSE);
	glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, &isCompiled);
	if (isCompiled == GL_FALSE)
	{
		GLint infoLogLength(0);
		glGetShaderiv(vertexShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);

		// The info log length includes the NULL character.
		std::vector<GLchar> infoLog(infoLogLength);
		glGetShaderInfoLog(vertexShaderID, infoLogLength, &infoLogLength, &infoLog[0]);

		// Display the info log message.
		std::cerr << &infoLog[0] << std::endl;

		// Don't need the shader anymore.
		glDeleteShader(vertexShaderID);

		exit(1);
	}

	// Compile the fragment shader.
	const GLchar* fragmentSource = (const GLchar*)fragmentShaderCode.c_str();
	glShaderSource(fragmentShaderID, 1, &fragmentSource, nullptr);
	glCompileShader(fragmentShaderID);

	// Check the fragment shader.
	glGetShaderiv(fragmentShaderID, GL_COMPILE_STATUS, &isCompiled);
	if (isCompiled == GL_FALSE)
	{
		GLint infoLogLength(0);
		glGetShaderiv(fragmentShaderID, GL_INFO_LOG_LENGTH, &infoLogLength);

		// The info log length includes the NULL character.
		std::vector<GLchar> infoLog(infoLogLength);
		glGetShaderInfoLog(fragmentShaderID, infoLogLength, &infoLogLength, &infoLog[0]);

		// Display the info log message.
		std::cerr << &infoLog[0] << std::endl;

		// Don't need the shaders anymore.
		glDeleteShader(vertexShaderID);
		glDeleteShader(fragmentShaderID);

		exit(1);
	}

	// Vertex and fragment shaders are successfully compiled.
	// Link them together into a program.

	GLuint programID = glCreateProgram();

	// Attach shaders to the program.
	glAttachShader(programID, vertexShaderID);
	glAttachShader(programID, fragmentShaderID);

	// Link the program.
	glLinkProgram(programID);


	// Check the program.
	GLint isLinked(GL_FALSE);
	glGetProgramiv(programID, GL_LINK_STATUS, &isLinked);
	if (isLinked == GL_FALSE)
	{
		GLint infoLogLength(0);
		glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &infoLogLength);

		// The info log length includes the NULL character.
		std::vector<GLchar> infoLog(infoLogLength);
		glGetProgramInfoLog(programID, infoLogLength, &infoLogLength, &infoLog[0]);

		// Display the info log message.
		std::cerr << &infoLog[0] << std::endl;

		// Don't need program anymore.
		glDeleteProgram(programID);
		// Don't need the shaders anymore.
		glDeleteShader(vertexShaderID);
		glDeleteShader(fragmentShaderID);

		exit(1);
	}

	// Detach shaders after linking.
	glDetachShader(programID, vertexShaderID);
	glDetachShader(programID, fragmentShaderID);

	// Only linking shaders to one program, so delete.
	glDeleteShader(vertexShaderID);
	glDeleteShader(fragmentShaderID);

	return programID;
}

GLuint gl_gen_texture2D(int sz)
{
	// Create an OpenGL texture.
	GLuint textureID;
	glGenTextures(1, &textureID);

	// Bind the texture. All future texture functions will modify this texture.
	glBindTexture(GL_TEXTURE_2D, textureID);
	// Give the image to OpenGL.
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, sz, sz, 0, GL_RED, GL_FLOAT, nullptr);

	// Poor filtering, or ...
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// Return the ID of the texture we just created
	return textureID;
}
