#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    
    // Elapse time to time out
    unsigned int timeOut;
    
    Timer();
    
    // Check if the timer has elapsed
    bool check();
    
    // Reset the timer
    void reset();
    
private:
    
    std::chrono::time_point<std::chrono::steady_clock> mLast;
};

#endif
