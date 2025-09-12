#include "GLContext.h"

#ifdef _WIN32

static LRESULT CALLBACK GLContext_DummyWndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

bool GLContext::createHiddenWindow() {
    hInst_ = GetModuleHandle(NULL);
    WNDCLASS wc = {};
    wc.style         = CS_OWNDC;
    wc.lpfnWndProc   = GLContext_DummyWndProc;
    wc.hInstance     = hInst_;
    wc.lpszClassName = TEXT("GLDummyClass");
    if (!RegisterClass(&wc)) {
        if (GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
            std::cerr << "GL: RegisterClass failed\n";
            return false;
        }
    }
    hWnd_ = CreateWindow(wc.lpszClassName, TEXT("gl"),
                         WS_OVERLAPPEDWINDOW,
                         CW_USEDEFAULT, CW_USEDEFAULT, 16, 16,
                         NULL, NULL, hInst_, NULL);
    if (!hWnd_) {
        std::cerr << "GL: CreateWindow failed\n";
        return false;
    }
    return true;
}

bool GLContext::setupLegacyPixelFormat() {
    hDC_ = GetDC(hWnd_);
    if (!hDC_) { std::cerr << "GL: GetDC failed\n"; return false; }
    PIXELFORMATDESCRIPTOR pfd = {};
    pfd.nSize      = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion   = 1;
    pfd.dwFlags    = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 24;
    pfd.cAlphaBits = 8;
    pfd.cDepthBits = 24;
    pfd.iLayerType = PFD_MAIN_PLANE;
    int pf = ChoosePixelFormat(hDC_, &pfd);
    if (pf == 0) { std::cerr << "GL: ChoosePixelFormat failed\n"; return false; }
    if (!SetPixelFormat(hDC_, pf, &pfd)) { std::cerr << "GL: SetPixelFormat failed\n"; return false; }
    return true;
}

bool GLContext::createModernOrFallbackContext() {
    // 1) Make a legacy context to load WGL extensions
    HGLRC legacy = wglCreateContext(hDC_);
    if (!legacy) { std::cerr << "GL: wglCreateContext (legacy) failed\n"; return false; }
    if (!wglMakeCurrent(hDC_, legacy)) { std::cerr << "GL: wglMakeCurrent (legacy) failed\n"; return false; }

    // 2) Initialize GLEW to load WGL entry points
    glewExperimental = GL_TRUE;
    (void)glewInit();

    // 3) Try to create a modern 4.3 core context
    PFNWGLCREATECONTEXTATTRIBSARBPROC wglCreateContextAttribsARB =
        (PFNWGLCREATECONTEXTATTRIBSARBPROC)wglGetProcAddress("wglCreateContextAttribsARB");

    int attribs[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
        WGL_CONTEXT_MINOR_VERSION_ARB, 3,
        WGL_CONTEXT_PROFILE_MASK_ARB,  WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
    #ifdef _DEBUG
        WGL_CONTEXT_FLAGS_ARB,         WGL_CONTEXT_DEBUG_BIT_ARB,
    #endif
        0
    };

    HGLRC modern = nullptr;
    if (wglCreateContextAttribsARB) {
        modern = wglCreateContextAttribsARB(hDC_, 0, attribs);
    }

    if (modern) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(legacy);
        hRC_ = modern;
        if (!wglMakeCurrent(hDC_, hRC_)) { std::cerr << "GL: wglMakeCurrent (modern) failed\n"; return false; }
    } else {
        // Fallback: keep legacy context
        hRC_ = legacy;
        if (!wglMakeCurrent(hDC_, hRC_)) { std::cerr << "GL: wglMakeCurrent (legacy reuse) failed\n"; return false; }
    }

    // 4) Final GLEW init on the context we're going to use
    glewExperimental = GL_TRUE;
    (void)glewInit();

    return true;
}

bool GLContext::init() {
    if (hRC_) return true; // already initialized
    if (!createHiddenWindow())            return false;
    if (!setupLegacyPixelFormat())        return false;
    if (!createModernOrFallbackContext()) return false;
    ShowWindow(hWnd_, SW_HIDE);
    const GLubyte* ver = glGetString(GL_VERSION);
    std::string verstr = (const char*)ver;
    verstr.erase(verstr.begin() + verstr.find(" "), verstr.end());
    std::cout << "OpenGL " << verstr << " GPU ready...\n";
    return true;
}

void GLContext::shutdown() {
    if (wglGetCurrentContext()) wglMakeCurrent(NULL, NULL);
    if (hRC_)  { wglDeleteContext(hRC_); hRC_ = nullptr; }
    if (hDC_ && hWnd_) { ReleaseDC(hWnd_, hDC_); hDC_ = nullptr; }
    if (hWnd_) { DestroyWindow(hWnd_); hWnd_ = nullptr; }
}

#else // !_WIN32

bool GLContext::init() { return false; }
void GLContext::shutdown() {}

#endif
