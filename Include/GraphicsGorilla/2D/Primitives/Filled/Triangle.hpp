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
		void DrawTriangle(float base, float height, glm::vec2 position, glm::vec4 color);
	  }
	}
  }
}