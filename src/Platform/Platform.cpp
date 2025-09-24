#include "Platform.h"

#include <conio.h>

void WindowResizePx(int width, int height) {
    HWND hwnd = GetConsoleWindow();
    if (!hwnd) return;
    RECT r;
    GetWindowRect(hwnd, &r);
    // keep current position, just change size
    MoveWindow(hwnd, r.left, r.top, width, height, TRUE);
}

SizePx DisplayGetSize() {
    return { GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN) };
}

bool KeyPressedNonBlocking() {
    return _kbhit() != 0;
}

int ReadKeyNonBlocking() {
    return _getch();
}

