#include "Timer.h"

Timer::Timer() : timeOut(60), mLast(std::chrono::steady_clock::now() + std::chrono::seconds(timeOut)) {}

bool Timer::check() {
    std::chrono::time_point<std::chrono::steady_clock> current = std::chrono::steady_clock::now();
    if (current >= mLast) {
        mLast = current + std::chrono::seconds(timeOut);
        return true;
    }
    return false;
}

void Timer::reset() {
    mLast = std::chrono::steady_clock::now() + std::chrono::seconds(timeOut);
}
