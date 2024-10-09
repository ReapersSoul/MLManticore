#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <ft2build.h>
#include <freetype/freetype.h>
#include <map>
#include <string>
#include <vector>

namespace GraphicsGorilla
{
	namespace TwoD
	{
		namespace Text
		{
			class Font
			{
			public:
				Font() = default;
				~Font();
				bool load(const std::string &fontPath, int size = 48);
				void render(const std::string &text, float x, float y, float scale, const glm::vec4 &color);
				void setPixelSize(int size);

			private:
				struct Character
				{
					GLuint textureID; // ID handle of the glyph texture
					int width;		  // Width of glyph
					int height;		  // Height of glyph
					int bearingX;	  // Offset from baseline to left of glyph
					int bearingY;	  // Offset from baseline to top of glyph
					GLuint advance;	  // Horizontal offset to advance to next glyph
				};

				std::map<GLchar, Character> characters;
				FT_Face face;
				FT_Library library;
			};
		}
	}
}