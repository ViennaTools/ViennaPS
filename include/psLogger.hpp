#pragma once

#include <iostream>

#include <psUtils.hpp>

// verbosity levels:
// 0 errors
// 1 + warnings
// 2 + info
// 3 + intermediate output
// 4 + timings
// 5 + debug
enum class psVerbosityEnum : unsigned {
  PS_VERB_ERROR = 0,
  PS_VERB_WARNING = 1,
  PS_VERB_INFO = 2,
  PS_VERB_TIMING = 3,
  PS_VERB_INTERMEDIATE = 4,
  PS_VERB_DEBUG = 5
};

psVerbosityEnum psVerbosity = psVerbosityEnum::PS_VERB_INFO;

/// Singleton class for thread-safe logging.
class psLogger {
  std::string message;

  bool error = false;
  const unsigned tabWidth = 4;

  psLogger() {}

public:
  // delete constructors to result in better error messages by compilers
  psLogger(const psLogger &) = delete;
  void operator=(const psLogger &) = delete;

  static void setVerbosity(const psVerbosityEnum verbLevel) {
    psVerbosity = verbLevel;
  }
  static unsigned getVerbosity() { return static_cast<unsigned>(psVerbosity); }

  static psLogger &getInstance() {
    static psLogger instance;
    return instance;
  }

  psLogger &addDebug(std::string s) {
    if (static_cast<unsigned>(psVerbosity) < 5)
      return *this;
#pragma omp critical
    { message += "\n" + std::string(tabWidth, ' ') + "DEBUG: " + s + "\n"; }
    return *this;
  }

  template <class Clock>
  psLogger &addTiming(std::string s, psUtils::Timer<Clock> &timer) {
    if (static_cast<unsigned>(psVerbosity) < 4)
      return *this;
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + s +
                 " took: " + std::to_string(timer.currentDuration * 1e-9) +
                 " s \n";
    }
    return *this;
  }

  psLogger &addInfo(std::string s) {
    if (static_cast<unsigned>(psVerbosity) < 2)
      return *this;
#pragma omp critical
    { message += std::string(tabWidth, ' ') + s + "\n"; }
    return *this;
  }

  psLogger &addWarning(std::string s) {
    if (static_cast<unsigned>(psVerbosity) < 1)
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
