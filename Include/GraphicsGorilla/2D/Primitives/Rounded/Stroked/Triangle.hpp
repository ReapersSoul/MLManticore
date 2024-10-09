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
		namespace Rounded{
	  namespace Stroked
	  {
		void DrawTriangle(float base, float height, float radius, float thickness, glm::vec2 position, glm::vec4 BGcolor, glm::vec4 OutlineColor);
	  }}
	}
  }
}