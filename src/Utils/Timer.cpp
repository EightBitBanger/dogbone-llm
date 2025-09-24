#include "Timer.h"

static inline std::chrono::minutes clamp_minutes(unsigned int m) {
    return std::chrono::minutes(std::max(1u, m));
}

Timer::Timer()
: timeOut(30),
  mPeriod(clamp_minutes(timeOut)),
  mDeadline(std::chrono::steady_clock::now() + mPeriod) {}

bool Timer::check() {
    const auto current = std::chrono::steady_clock::now();
    if (current >= mDeadline) {
        // Keep a steady cadence relative to the previous deadline.
        mDeadline += mPeriod;
        // If multiple periods were missed, resync to avoid runaway loops.
        if (current >= mDeadline) {
            mDeadline = current + mPeriod;
        }
        return true;
    }
    return false;
}

void Timer::reset() {
    // Recompute cached period from (possibly updated) timeOut and restart from now.
    mPeriod   = clamp_minutes(timeOut);
    mDeadline = std::chrono::steady_clock::now() + mPeriod;
}


std::string Timer::GetDate(void) {
    std::time_t t = std::time(NULL);
    std::tm local_tm;
#if defined(_WIN32)
    localtime_s(&local_tm, &t);
#else
    localtime_r(&t, &local_tm);
#endif
    char buf[32];
    // YYYYMMDD-HHMM (zero-padded)
    std::strftime(buf, sizeof(buf), "%Y-%m", &local_tm);
    std::string name = std::string(buf);
    name.erase(name.begin(), name.begin() + 2);
    return name;
}
