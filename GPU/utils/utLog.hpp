#pragma once

#include <cuda.h>
#include <iostream>

#ifndef PRINT
#define PRINT(var) std::cout << #var << "=" << var << std::endl;
#define PING                                                                   \
  std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__            \
            << std::endl;
#endif

#define UT_NOTIMPLEMENTED                                                      \
  throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +                  \
                           " not implemented")

#define UT_TERMINAL_RED "\033[1;31m"
#define UT_TERMINAL_GREEN "\033[1;32m"
#define UT_TERMINAL_YELLOW "\033[1;33m"
#define UT_TERMINAL_BLUE "\033[1;34m"
#define UT_TERMINAL_RESET "\033[0m"
#define UT_TERMINAL_DEFAULT UT_TERMINAL_RESET
#define UT_TERMINAL_BOLD "\033[1;1m"

/// Singleton class for thread-safe logging.
class utLog {
  std::string message;

  bool error = false;
  const unsigned tabWidth = 4;

  utLog() {}

  std::string getErrorString(CUresult err) {
    const char *errorMsg[2048];
    cuGetErrorString(err, errorMsg);
    std::string errorString = *errorMsg;
    return errorString;
  }

public:
  // delete constructors to result in better error messages by compilers
  utLog(const utLog &) = delete;
  void operator=(const utLog &) = delete;

  static utLog &getInstance() {
    static utLog instance;
    return instance;
  }

  utLog &add(std::string s) {
#pragma omp critical
    { message += "\n" + std::string(tabWidth, ' ') + "WARNING: " + s + "\n"; }
    return *this;
  }

  utLog &addWarning(std::string s) {
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + UT_TERMINAL_YELLOW +
                 "WARNING: " + s + "\n";
    }
    return *this;
  }

  utLog &addError(std::string s, bool shouldAbort = true) {
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + UT_TERMINAL_RED +
                 "ERROR: " + s + "\n";
      // always abort once error message should be printed
      error = true;
    }
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }

  utLog &addModuleError(std::string moduleName, CUresult err,
                        bool shouldAbort = true) {
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + UT_TERMINAL_RED +
                 "ERROR in CUDA module " + moduleName + ": " +
                 getErrorString(err) + "\n";
      // always abort once error message should be printed
      error = true;
    }
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }

  utLog &addFunctionError(std::string kernelName, CUresult err,
                          bool shouldAbort = true) {
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + UT_TERMINAL_RED +
                 "ERROR in CUDA kernel " + kernelName + ": " +
                 getErrorString(err) + "\n";
      // always abort once error message should be printed
      error = true;
    }
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }

  utLog &addLaunchError(std::string kernelName, CUresult err,
                        bool shouldAbort = true) {
#pragma omp critical
    {
      message += "\n" + std::string(tabWidth, ' ') + UT_TERMINAL_RED +
                 "ERROR in CUDA kernel launch (" + kernelName +
                 "): " + getErrorString(err) + "\n";
      // always abort once error message should be printed
      error = true;
    }
    // abort now if asked
    if (shouldAbort)
      print();
    return *this;
  }

  utLog &addDebug(std::string s) {
#pragma omp critical
    {
      message +=
          UT_TERMINAL_GREEN + std::string(tabWidth, ' ') + "DEBUG: " + s + "\n";
    }
    return *this;
  }

  void print(std::ostream &out = std::cout) {
#pragma omp critical
    {
      out << message;
      out << UT_TERMINAL_DEFAULT;
      message.clear();
      if (error)
        abort();
    }
  }
};
