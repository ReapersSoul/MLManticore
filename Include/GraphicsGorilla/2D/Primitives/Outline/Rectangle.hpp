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
	  namespace Outline
	  {
		void DrawRectangle(float width, float height, float thickness, glm::vec2 position, glm::vec4 color);
	  }
	}
  }
}