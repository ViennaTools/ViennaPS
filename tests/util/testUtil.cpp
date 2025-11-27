#include <iostream>
#include <psUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennacore::util;

void TestIntegrationSchemeConversion() {
  // Test string to enum
  VC_TEST_ASSERT(convertIntegrationScheme("ENGQUIST_OSHER_1ST_ORDER") ==
                 viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);
  VC_TEST_ASSERT(convertIntegrationScheme("EO_1") ==
                 viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);

  // Test enum to string
  VC_TEST_ASSERT(
      convertIntegrationSchemeToString(
          viennals::IntegrationSchemeEnum::ENGQUIST_OSHER_1ST_ORDER) ==
      "ENGQUIST_OSHER_1ST_ORDER");

  // Test round trip
  auto scheme = viennals::IntegrationSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  VC_TEST_ASSERT(convertIntegrationScheme(
                     convertIntegrationSchemeToString(scheme)) == scheme);
}

void TestBoundaryConditionConversion() {
  VC_TEST_ASSERT(convertBoundaryCondition(
                     viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY) ==
                 viennaray::BoundaryCondition::REFLECTIVE);
  VC_TEST_ASSERT(convertBoundaryCondition(
                     viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY) ==
                 viennaray::BoundaryCondition::PERIODIC);
}

void TestMetaDataToString() {
  std::unordered_map<std::string, std::vector<double>> metaData;
  metaData["key1"] = {1.0, 2.0};
  std::string str = metaDataToString(metaData);
  VC_TEST_ASSERT(str.find("key1") != std::string::npos);
  VC_TEST_ASSERT(str.find("1.000000") != std::string::npos);
}

int main() {
  TestIntegrationSchemeConversion();
  TestBoundaryConditionConversion();
  TestMetaDataToString();
  std::cout << "Util tests passed!" << std::endl;
  return 0;
}
