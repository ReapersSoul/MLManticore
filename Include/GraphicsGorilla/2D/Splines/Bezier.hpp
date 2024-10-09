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
	    void DrawBezier(glm::vec2 start, glm::vec2 control1, glm::vec2 control2, glm::vec2 end, glm::vec4 color);
	  }
	}
}