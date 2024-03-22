#include <psLogger.hpp>
#include <psTestAssert.hpp>

int main() {
  psLogger &logger = psLogger::getInstance();

  std::stringstream ss;

  logger.setLogLevel(psLogLevel::TIMING);
  PSTEST_ASSERT(logger.getLogLevel() == 3);

  logger.setLogLevel(psLogLevel::DEBUG);
  logger.addDebug("Debug message");
  logger.print(ss);

  PSTEST_ASSERT(ss.str() == "    DEBUG: Debug message\n");
  ss.str("");

  logger.setLogLevel(psLogLevel::TIMING);
  logger.addTiming("Timing message", 1.23);
  logger.print(ss);

  PSTEST_ASSERT(ss.str().find("    Timing message: 1.23") == 0);
  ss.str("");

  logger.setLogLevel(psLogLevel::INFO);
  logger.addInfo("Info message");
  logger.print(ss);

  PSTEST_ASSERT(ss.str() == "    Info message\n");
  ss.str("");

  logger.setLogLevel(psLogLevel::WARNING);
  logger.addWarning("Warning message");
  logger.print(ss);

  PSTEST_ASSERT(ss.str() == "\n    WARNING: Warning message\n");
  ss.str("");
}
