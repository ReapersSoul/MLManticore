#pragma once
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <glm/gtc/type_ptr.hpp>
#include <functional>

namespace GraphicsGorilla
{

  class FrameBuffer
  {
    unsigned int ID;
    unsigned int textureColorBuffer;
    unsigned int RBO;
    int LastBuffer;
    int width, height;

  public:
    void DrawToFB(glm::vec4 BackgroundColor,std::function<void()> drawFunction);

    FrameBuffer();
    
    bool Init(int width, int height);

    ~FrameBuffer();

    void Bind();

    void Unbind();

    unsigned int GetTexture();

    glm::vec2 GetSize();

    void DrawFBToScreen(glm::vec2 position, glm::vec2 size,glm::vec4 tint={1.0f,1.0f,1.0f,1.0f});
  };

}