#ifndef GL_CONTEXT_H
#define GL_CONTEXT_H

#include <iostream>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/wglew.h>


class GLContext {
public:
    GLContext() = default;
    ~GLContext() { shutdown(); }

    GLContext(const GLContext&) = delete;
    GLContext& operator=(const GLContext&) = delete;
    GLContext(GLContext&&) = delete;
    GLContext& operator=(GLContext&&) = delete;

    // Create and make current a GL context. Returns true on success.
    bool init();

    // Destroy context/window if created.
    void shutdown();

#ifdef _WIN32
    HDC   dc() const { return hDC_; }
    HGLRC rc() const { return hRC_; }
    HWND  wnd() const { return hWnd_; }
#endif

private:
#ifdef _WIN32
    bool createHiddenWindow();
    bool setupLegacyPixelFormat();
    bool createModernOrFallbackContext();
#endif

private:
#ifdef _WIN32
    HINSTANCE hInst_ = nullptr;
    HWND      hWnd_  = nullptr;
    HDC       hDC_   = nullptr;
    HGLRC     hRC_   = nullptr;
#endif
};

#endif // GL_CONTEXT_H
