#include <psLogger.hpp>
#include <psTestAssert.hpp>

int main() {
  psLogger &logger = psLogger::getInstance();

  // Redirect std::cout to a stringstream to capture logger output
  std::stringstream ss;
  std::streambuf *oldCoutStreamBuf = std::cout.rdbuf();

  std::cout.rdbuf(ss.rdbuf()); // redirect cout to ss

  logger.setLogLevel(psLogLevel::TIMING);
  PSTEST_ASSERT(logger.getLogLevel() == 3);

  logger.setLogLevel(psLogLevel::DEBUG);
  logger.addDebug("Debug message");
  PSTEST_ASSERT(ss.str() == "    DEBUG: Debug message\n");

  logger.setLogLevel(psLogLevel::TIMING);
  logger.addTiming("Timing message", 1.23);
  PSTEST_ASSERT(ss.str() == "    Timing message: 1.23 s \n");

  logger.setLogLevel(psLogLevel::INFO);
  logger.addInfo("Info message");
  PSTEST_ASSERT(ss.str() == "    Info message\n");

  logger.setLogLevel(psLogLevel::WARNING);
  logger.addWarning("Warning message");
  PSTEST_ASSERT(ss.str() == "\n    WARNING: Warning message\n");

  std::cout.rdbuf(oldCoutStreamBuf); // redirect cout back to old buffer
}
