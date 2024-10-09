#pragma once
#include <glm/glm.hpp>

namespace GraphicsGorilla{
	namespace Color{
    typedef glm::vec4 Color;

    const glm::vec4 Red = {1.0f, 0.0f, 0.0f, 1.0f};
    const glm::vec4 Green = {0.0f, 1.0f, 0.0f, 1.0f};
    const glm::vec4 Blue = {0.0f, 0.0f, 1.0f, 1.0f};
    const glm::vec4 Yellow = {1.0f, 1.0f, 0.0f, 1.0f};
    const glm::vec4 Cyan = {0.0f, 1.0f, 1.0f, 1.0f};
    const glm::vec4 Magenta = {1.0f, 0.0f, 1.0f, 1.0f};
    const glm::vec4 White = {1.0f, 1.0f, 1.0f, 1.0f};
    const glm::vec4 Black = {0.0f, 0.0f, 0.0f, 1.0f};

    glm::vec4 Lerp(const glm::vec4& a, const glm::vec4& b, float t);

    glm::vec4 Random();

    glm::vec4 Random(float alpha);

    glm::vec4 RGBAtoHSVA(const glm::vec4& color);

    glm::vec4 HSVAtoRGBA(const glm::vec4& color);

    glm::vec4 RGBA(float r, float g, float b, float a);

    glm::vec4 HSVA(float h, float s, float v, float a);

    float map(float value, float start1, float stop1, float start2, float stop2);
  }
}