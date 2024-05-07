#pragma once

#include "csUtil.hpp"

#include <string>

// verbosity levels:
// 0 errors
// 1 + warnings
// 2 + info
// 3 + timings
// 4 + intermediate output (meshes)
// 5 + debug
enum class csLogLevel : unsigned {
  ERROR = 0,
  WARNING = 1,
  INFO = 2,
  TIMING = 3,
  INTERMEDIATE = 4,
  DEBUG = 5
};

/// Singleton class for thread-safe logging. The logger can be accessed via
/// csLogger::getInstance(). The logger can be configured to print messages of a
/// certain level or lower. The default level is INFO. The different logging
/// levels are: ERROR, WARNING, INFO, TIMING, INTERMEDIATE, DEBUG. The logger
/// can also be used to print timing information.
class csLogger {
  std::string message;

  bool error = false;
  const unsigned tabWidth = 4;
  static csLogLevel logLevel;

  csLogger() {}

public:
  // delete constructors to result in better error messages by compilers
  csLogger(const csLogger &) = delete;
  void operator=(const csLogger &) = delete;

  // Set the log level for all instances of the logger.
  static void setLogLevel(const csLogLevel passedLogLevel) {
    logLevel = passedLogLevel;
  }

  static unsigned getLogLevel() { return static_cast<unsigned>(logLevel); }

  static csLogger &getInstance() {
    static csLogger instance;
    return instance;
  }

  // Add debug message if log level is high enough.
  csLogger &addDebug(std::string s) {
    if (getLogLevel() < 5)
      return *this;
#pragma omp critical
    { message += std::string(tabWidth, ' ') + "DEBUG: " + s + "\n"; }
    return *this;
  }

  // Add timing message if log level is high enough.
  template <class Clock>
  csLogger &addTiming(std::string s, csUtil::Timer<Clock> &timer) {
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

  csLogger &addTiming(std::string s, double timeInSeconds) {
    if (getLogLevel() < 3)
      return *this;
#pragma omp critical
    {
      message += std::string(tabWidth, ' ') + s + ": " +
                 std::to_string(timeInSeconds) + " s \n";
    }
    return *this;
  }

  csLogger &addTiming(std::string s, double timeInSeconds,
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

  // Add info message if log level is high enough.
  csLogger &addInfo(std::string s) {
    if (getLogLevel() < 2)
      return *this;
#pragma omp critical
    { message += std::string(tabWidth, ' ') + s + "\n"; }
    return *this;
  }

  // Add warning message if log level is high enough.
  csLogger &addWarning(std::string s) {
    if (getLogLevel() < 1)
      return *this;
#pragma omp critical
    { message += "\n" + std::string(tabWidth, ' ') + "WARNING: " + s + "\n"; }
    return *this;
  }

  // Add error message if log level is high enough.
  csLogger &addError(std::string s, bool shouldAbort = true) {
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

  // Print message to std::cout if log level is high enough.
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
inline csLogLevel csLogger::logLevel = csLogLevel::INTERMEDIATE;
