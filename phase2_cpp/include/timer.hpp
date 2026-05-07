#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <stdexcept>

/// Per-phase wall-clock timer using std::chrono::steady_clock.
///
/// Usage:
///   PerPhaseTimer t;
///   t.start("qr");
///   ...  // Eigen/BLAS calls
///   t.stop("qr");
///   double secs = t.get("qr");  // seconds
///
/// Nested or interleaved phases are NOT supported — each phase is a simple
/// start/stop pair.  Calling start() twice without stop() throws.
class PerPhaseTimer {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /// Begin timing a named phase.
    void start(const std::string& phase) {
        if (active_.count(phase))
            throw std::logic_error("PerPhaseTimer: start() called twice for phase '" + phase + "'");
        active_[phase] = Clock::now();
    }

    /// End timing a named phase and accumulate the elapsed time.
    void stop(const std::string& phase) {
        auto it = active_.find(phase);
        if (it == active_.end())
            throw std::logic_error("PerPhaseTimer: stop() called without start() for phase '" + phase + "'");
        double elapsed = std::chrono::duration<double>(Clock::now() - it->second).count();
        accumulated_[phase] += elapsed;
        active_.erase(it);
    }

    /// Return accumulated seconds for a phase (0.0 if never timed).
    double get(const std::string& phase) const {
        auto it = accumulated_.find(phase);
        return (it != accumulated_.end()) ? it->second : 0.0;
    }

    /// Return the full accumulated map (phase → seconds).
    const std::unordered_map<std::string, double>& all() const {
        return accumulated_;
    }

    /// Reset all timings.
    void reset() {
        active_.clear();
        accumulated_.clear();
    }

private:
    std::unordered_map<std::string, TimePoint> active_;
    std::unordered_map<std::string, double>    accumulated_;
};
