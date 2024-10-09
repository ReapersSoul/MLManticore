#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace GraphicsGorilla
{
  namespace TwoD
  {
	namespace Primitives
	{
	  namespace Filled
	  {
		void DrawCircle(float radius, int segments, glm::vec2 position, glm::vec4 color);
	  }
	}
  }
}