#include <cmath>
#include <iostream>
#include <psUnits.hpp>
#include <vcTestAsserts.hpp>

using namespace viennaps::units;

// Helper for float comparison
bool approxEqual(double a, double b, double epsilon = 1e-6) {
  return std::abs(a - b) < epsilon;
}

void TestLength() {
  // Test default
  Length::setUnit(Length::UNDEFINED);
  VC_TEST_ASSERT(Length::getUnit() == Length::UNDEFINED);

  // Test setting by string
  Length::setUnit("meter");
  VC_TEST_ASSERT(Length::getUnit() == Length::METER);
  Length::setUnit("nm");
  VC_TEST_ASSERT(Length::getUnit() == Length::NANOMETER);

  // Test conversions
  Length::setUnit(Length::METER);
  VC_TEST_ASSERT(approxEqual(Length::convertMeter(), 1.0));
  VC_TEST_ASSERT(approxEqual(Length::convertCentimeter(), 100.0));

  Length::setUnit(Length::NANOMETER);
  VC_TEST_ASSERT(approxEqual(Length::convertMeter(), 1e-9));
  VC_TEST_ASSERT(approxEqual(Length::convertNanometer(), 1.0));
}

void TestTime() {
  // Test default
  Time::setUnit(Time::UNDEFINED);
  VC_TEST_ASSERT(Time::getUnit() == Time::UNDEFINED);

  // Test setting by string
  Time::setUnit("second");
  VC_TEST_ASSERT(Time::getUnit() == Time::SECOND);
  Time::setUnit("ms");
  VC_TEST_ASSERT(Time::getUnit() == Time::MILLISECOND);

  // Test conversions
  Time::setUnit(Time::SECOND);
  VC_TEST_ASSERT(approxEqual(Time::convertSecond(), 1.0));
  VC_TEST_ASSERT(approxEqual(Time::convertMillisecond(), 1000.0));
}

int main() {
  TestLength();
  TestTime();
  std::cout << "Units tests passed!" << std::endl;
  return 0;
}
