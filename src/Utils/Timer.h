#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <chrono>
#include <algorithm>

class Timer {
public:
    
    unsigned int timeOut;
    
    Timer();
    
    // Returns true when the timer elapses and advances the deadline.
    bool check();
    
    // Recompute period from timeOut and start a fresh countdown from now.
    void reset();
    
    std::string GetDate(void);
    
private:
    // Cached period as a chrono duration, derived from timeOut.
    std::chrono::minutes mPeriod{std::chrono::minutes(30)};
    
    // The next time point when the timer should fire.
    std::chrono::time_point<std::chrono::steady_clock> mDeadline;
};


#endif
