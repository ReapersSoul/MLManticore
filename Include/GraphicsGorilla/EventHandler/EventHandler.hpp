#pragma once

namespace GraphicsGorilla
{
  class EventHandler
  {
  public:
	virtual void KeyPressed(int key, int scancode, int action, int mods) = 0;
	virtual void MouseMoved(double xpos, double ypos) = 0;
	virtual void MouseButtonPressed(int button, int action, int mods) = 0;
	virtual void MouseScrolled(double xoffset, double yoffset) = 0;
  };
}