#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <functional>

namespace GraphicsGorilla
{
  class Window
  {
    GLFWwindow *window;
    std::function<void()> Init;
    std::function<void(int,int)> Draw;
    std::function<void(GLFWwindow *)> Update;
    std::function<void(GLFWwindow *, int, int, int, int)> KeyCallback;
    std::function<void(GLFWwindow *, int, int, int)> MouseButtonCallback;
    std::function<void(GLFWwindow *, double, double)> MouseMoveCallback;
    std::function<void(GLFWwindow *, double, double)> MouseScrollCallback;
    std::function<void(GLFWwindow *, int, int)> ControllerCallback;

    //callbacks
    static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
    {
      Window *w = (Window *)glfwGetWindowUserPointer(window);
      if (w){
        if (w->GetKeyCallback())
        {
          w->GetKeyCallback()(window, key, scancode, action, mods);
        }
      }
    }

    static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
    {
      Window *w = (Window *)glfwGetWindowUserPointer(window);
      if (w){
        if(w->GetMouseButtonCallback())
        {
          w->GetMouseButtonCallback()(window, button, action, mods);
        }
      }
    }

    static void mouse_move_callback(GLFWwindow *window, double xpos, double ypos)
    {
      Window *w = (Window *)glfwGetWindowUserPointer(window);
      if (w){
        if(w->GetMouseMoveCallback())
        {
          w->GetMouseMoveCallback()(window, xpos, ypos);
        }
      }
    }

    static void mouse_scroll_callback(GLFWwindow *window, double xoffset, double yoffset)
    {
      Window *w = (Window *)glfwGetWindowUserPointer(window);
      if (w){
        if(w->GetMouseScrollCallback())
        {
          w->GetMouseScrollCallback()(window, xoffset, yoffset);
        }
      }
    }

    static void controller_callback(int jid, int event)
    {
      Window *w = (Window *)glfwGetWindowUserPointer(glfwGetCurrentContext());
      if (w){
        if(w->GetControllerCallback())
        {
          w->GetControllerCallback()(glfwGetCurrentContext(), jid, event);
        }
      }
    }

    //callback getters
    std::function<void(GLFWwindow *, int, int, int, int)> GetKeyCallback()
    {
      return KeyCallback;
    }

    std::function<void(GLFWwindow *, int, int, int)> GetMouseButtonCallback()
    {
      return MouseButtonCallback;
    }

    std::function<void(GLFWwindow *, double, double)> GetMouseMoveCallback()
    {
      return MouseMoveCallback;
    }

    std::function<void(GLFWwindow *, double, double)> GetMouseScrollCallback()
    {
      return MouseScrollCallback;
    }

    std::function<void(GLFWwindow *, int, int)> GetControllerCallback()
    {
      return ControllerCallback;
    }

  public:
    Window(int width, int height, const char *title)
    {
      if (!glfwInit())
      {
        return;
      }
      window = glfwCreateWindow(width, height, title, NULL, NULL);
      if (!window)
      {
        glfwTerminate();
        return;
      }
      glfwMakeContextCurrent(window);
      if (glewInit() != GLEW_OK)
      {
        return;
      }
      glfwSetWindowUserPointer(window, this);
    }

    ~Window()
    {
      glfwDestroyWindow(window);
      glfwTerminate();
    }

    void Run()
    {
      while (!glfwWindowShouldClose(window))
      {
        if (Update)
        {
          Update(window);
        }
        if (Draw)
        {
          int width, height;
          glfwGetFramebufferSize(window, &width, &height);
          Draw(width, height);
        }
        glfwSwapBuffers(window);
        glfwPollEvents();
      }
    }

    void SetDrawFunction(std::function<void(int,int)> draw)
    {
      Draw = draw;
    }

    void SetUpdateFunction(std::function<void(GLFWwindow *)> update)
    {
      Update = update;
    }

    void SetKeyCallback(std::function<void(GLFWwindow *, int, int, int, int)> _key_callback)
    {
      KeyCallback = _key_callback;
      glfwSetKeyCallback(window, key_callback);
    }

    void SetMouseButtonCallback(std::function<void(GLFWwindow *, int, int, int)> _mouse_button_callback)
    {
      MouseButtonCallback = _mouse_button_callback;
      glfwSetMouseButtonCallback(window, mouse_button_callback);
    }

    void SetMouseMoveCallback(std::function<void(GLFWwindow *, double, double)> _mouse_move_callback)
    {
      MouseMoveCallback = _mouse_move_callback;
      glfwSetCursorPosCallback(window, mouse_move_callback);
    }

    void SetMouseScrollCallback(std::function<void(GLFWwindow *, double, double)> _mouse_scroll_callback)
    {
      MouseScrollCallback = _mouse_scroll_callback;
      glfwSetScrollCallback(window, mouse_scroll_callback);
    }

    void SetControllerCallback(std::function<void(GLFWwindow *, int, int)> _controller_callback)
    {
      ControllerCallback = _controller_callback;
      glfwSetJoystickCallback(controller_callback);
    }
  };
}