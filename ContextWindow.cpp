#include "ContextWindow.h"

ContextWindow::ContextWindow(unsigned int maxSize) : maxSize(maxSize) {}

void ContextWindow::Add(int token) {
    if (tokens.size() > maxSize) 
        tokens.erase(tokens.begin());
    tokens.emplace(tokens.end(), token);
}

std::vector<int>& ContextWindow::GetContext() {
    return tokens;
}

void ContextWindow::Clear() {
    tokens.clear();
}

unsigned int ContextWindow::Size() const {
    return tokens.size();
}

unsigned int ContextWindow::Capacity() const {
    return maxSize;
}
