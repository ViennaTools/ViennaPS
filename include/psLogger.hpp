#pragma once

#include <iostream>

#include <psUtils.hpp>

// verbosity levels:
// 0 errors
// 1 + warnings
// 2 + info
// 3 + intermediate output (meshes)
// 4 + timings
// 5 + debug
enum class psLogLevel : unsigned {
  ERROR = 0,
  WARNING = 1,
  INFO = 2,
  TIMING = 3,
  INTERMEDIATE = 4,
  DEBUG = 5
};

/// Singleton class for thread-safe logging.
class psLogger {
  std::string message;

  bool error = false;
  const unsigned tabWidth = 4;
  static psLogLevel logLevel;

  psLogger() {}

public:
  // delete constructors to result in better error messages by compilers
  psLogger(const psLogger &) = delete;
  void operator=(const psLogger &) = delete;

  static void setLogLevel(const psLogLevel passedLogLevel) {
    logLevel = passedLogLevel;
  }

  static unsigned getLogLevel() { return static_cast<unsigned>(logLevel); }

  static psLogger &getInstance() {
    static psLogger instance;
    return instance;
  }

  psLogger &addDebug(std::string s) {
    if (getLogLevel() < 5)
      return *this;
#pragma omp critical
    { message += std::string(tabWidth, ' ') + "DEBUG: " + s + "\n"; }
    return *this;
  }

  template <class Clock>
  psLogger &addTiming(std::string s, psUtils::Timer<Clock> &timer) {
    if (getLogLevel() < 3)
      return *this;
#pragma omp critical
    {
      message += std::string(tabWidth, ' ') + s +
                 " took: " + std::to_string(timer.currentDuration * 1e-9) +
                 " s \n";
    }
    return *this;
  }

  psLogger &addTiming(std::string s, double timeInSeconds) {
    if (getLogLevel() < 3)
      return *this;
#pragma omp critical
    {
      message += std::string(tabWidth, ' ') + s + ": " +
                 std::to_string(timeInSeconds) + " s \n";
    }
    return *this;
  }

  psLogger &addTiming(std::string s, double timeInSeconds,
                      double totalTimeInSeconds) {
    if (getLogLevel() < 3)
      return *this;
#pragma omp critical
    {
      message += std::string(tabWidth, ' ') + s + ": " +
                 std::to_string(timeInSeconds) + " s\n" +
                 std::string(tabWidth, ' ') + "Percent of total time: " +
                 std::to_string(timeInSeconds / totalTimeInSeconds * 100) +
                 "\n";
    }
    return *this;
  }

  psLogger &addInfo(std::string s) {
    if (getLogLevel() < 2)
      return *this;
#pragma omp critical
    { message += std::string(tabWidth, ' ') + s + "\n"; }
    return *this;
  }

  psLogger &addWarning(std::string s) {
    if (getLogLevel() < 1)
      return *this;
#pragma omp critical
    { message += "\n" + std::string(tabWidth, ' ') + "WARNING: " + s + "\n"; }
    return *this;
  }

  psLogger &addError(std::string s, bool shouldAbort = true) {
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + "ERROR: " + s + "\n";
      // always abort once error message should be printed
      error = true;
    }
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }

  void print(std::ostream &out = std::cout) {
#pragma omp critical
    {
      out << message;
      message.clear();
      if (error)
        abort();
    }
  }
};

// initialize static member of logger
psLogLevel psLogger::logLevel = psLogLevel::INFO;
