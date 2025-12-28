#include <iostream>
#include <psUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennacore::util;

void TestSpatialSchemeConversion() {
  // Test string to enum
  VC_TEST_ASSERT(convertSpatialScheme("ENGQUIST_OSHER_1ST_ORDER") ==
                 viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);
  VC_TEST_ASSERT(convertSpatialScheme("EO_1") ==
                 viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);

  // Test string to enum
  VC_TEST_ASSERT(convertSpatialScheme("WENO_5TH_ORDER") ==
                 viennals::SpatialSchemeEnum::WENO_5TH_ORDER);
  VC_TEST_ASSERT(convertSpatialScheme("WENO_5") ==
                 viennals::SpatialSchemeEnum::WENO_5TH_ORDER);

  // Test enum to string
  VC_TEST_ASSERT(convertSpatialSchemeToString(
                     viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER) ==
                 "ENGQUIST_OSHER_1ST_ORDER");

  // Test round trip
  auto scheme = viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  VC_TEST_ASSERT(convertSpatialScheme(convertSpatialSchemeToString(scheme)) ==
                 scheme);
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
  std::cout << "Meta data str: " << str << std::endl;
  VC_TEST_ASSERT(str.find("key1") != std::string::npos);
  VC_TEST_ASSERT(str.find("1") != std::string::npos);
}

int main() {
  TestSpatialSchemeConversion();
  TestBoundaryConditionConversion();
  TestMetaDataToString();
  std::cout << "Util tests passed!" << std::endl;
  return 0;
}
