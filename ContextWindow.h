#ifndef CONTEXT_WINDOW_H
#define CONTEXT_WINDOW_H

#include <vector>

class ContextWindow {
public:
    
    void Add(int token);
    
    std::vector<int>& GetContext();
    
    void Clear();
    
    unsigned int Size() const;
    unsigned int Capacity() const;
    
    ContextWindow(unsigned int maxSize);
    
    int operator[](size_t i) {
        if (i >= tokens.size()) 
            return -1;
        return tokens[i];
    }
    
private:
    
    size_t maxSize;
    std::vector<int> tokens;
};

#endif
