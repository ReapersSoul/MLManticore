#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <ft2build.h>
#include <freetype/freetype.h>
#include <map>
#include <string>


#include "2D/2D.hpp"
#include "3D/3D.hpp"
#include "Color/Color.hpp"
#include "Framebuffer/FrameBuffer.hpp"
#include "Window/Window.hpp"

namespace GraphicsGorilla
{
  float map(float value, float start1, float stop1, float start2, float stop2);

  namespace Primitives
  {
    namespace TwoD
    {
      namespace Filled
      {
        void DrawTriangle(float base, float height, glm::vec2 position, glm::vec4 color);

        void DrawRectangle(float width, float height, glm::vec2 position, glm::vec4 color);

        void DrawCircle(float radius, int segments, glm::vec2 position, glm::vec4 color);

        namespace Rounded
        {

          void DrawTriangle(float base, float height, float radius, glm::vec2 position, glm::vec4 color);

          void DrawRectangle(float width, float height, float radius, glm::vec2 position, glm::vec4 color);
        }
      }

      namespace Outline
      {
        void DrawTriangle(float base, float height, float line_thickness, glm::vec2 position, glm::vec4 color);

        void DrawRectangle(float width, float height, float line_thickness, glm::vec2 position, glm::vec4 color);

        void DrawCircle(float radius, int segments, float line_thickness, glm::vec2 position, glm::vec4 color);

        namespace Rounded
        {
          void DrawTriangle(float base, float height, float radius, float line_thickness, glm::vec2 position, glm::vec4 color);

          void DrawRectangle(float width, float height, float radius, float line_thickness, glm::vec2 position, glm::vec4 color);

        }
      }

      namespace Stroked
      {
        void DrawTriangle(float base, float height, float line_thickness, glm::vec2 position, glm::vec4 FillColor, glm::vec4 StrokeColor);

        void DrawRectangle(float width, float height, float line_thickness, glm::vec2 position, glm::vec4 FillColor, glm::vec4 StrokeColor);

        void DrawCircle(float radius, int segments, float line_thickness, glm::vec2 position, glm::vec4 FillColor, glm::vec4 StrokeColor);
      
        namespace Rounded
        {
          void DrawTriangle(float base, float height, float radius, float line_thickness, glm::vec2 position, glm::vec4 FillColor, glm::vec4 StrokeColor);

          void DrawRectangle(float width, float height, float radius, float line_thickness, glm::vec2 position, glm::vec4 FillColor, glm::vec4 StrokeColor);
        }
      }

      namespace Splines
      {
        void DrawQuadraticBezier(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, int segments, glm::vec4 color);

        void DrawCubicBezier(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec2 p3, int segments, glm::vec4 color);

        void DrawCatmullRomSpline(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, glm::vec2 p3, int segments, glm::vec4 color);

        void DrawBezierSpline(std::vector<glm::vec2> points, int segments, glm::vec4 color);
      }

      
    }
    namespace ThreeD
    {
      void DrawPyramid(float base, float height, glm::vec4 position, glm::vec4 color);

      void DrawRectangularPrism(float width, float height, float depth, glm::vec4 position, glm::vec4 color);

      void DrawSphere(float radius, int segments, glm::vec4 position, glm::vec4 color);
    }
  }


  
}
