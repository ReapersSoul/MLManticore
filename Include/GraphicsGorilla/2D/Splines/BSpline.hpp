#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace GraphicsGorilla
{
	namespace TwoD
	{
	  namespace Splines
	  {
	    void DrawBSpline(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec2 p3, glm::vec4 color);
	  }
	}
}