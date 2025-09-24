#ifndef PLATFORM_LAYER_H
#define PLATFORM_LAYER_H

#include <windows.h>

// Windowing

struct SizePx { int width, height; };
void WindowResizePx(int width, int height);
SizePx DisplayGetSize();

// Input

bool KeyPressedNonBlocking();
int ReadKeyNonBlocking();

#endif
