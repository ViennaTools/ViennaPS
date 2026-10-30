#include <iostream>
#include <psUtil.hpp>
#include <vcTestAsserts.hpp>

using namespace viennacore::util;

void TestSpatialSchemeConversion() {
  // Test string to enum
  VC_TEST_ASSERT(detail::convertSpatialScheme("ENGQUIST_OSHER_1ST_ORDER") ==
                 viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);
  VC_TEST_ASSERT(detail::convertSpatialScheme("EO_1") ==
                 viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);

  // Test string to enum
  VC_TEST_ASSERT(detail::convertSpatialScheme("WENO_5TH_ORDER") ==
                 viennals::SpatialSchemeEnum::WENO_5TH_ORDER);
  VC_TEST_ASSERT(detail::convertSpatialScheme("WENO_5") ==
                 viennals::SpatialSchemeEnum::WENO_5TH_ORDER);

  // Test enum to string
  VC_TEST_ASSERT(detail::convertSpatialSchemeToString(
                     viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER) ==
                 "ENGQUIST_OSHER_1ST_ORDER");

  // Test round trip
  auto scheme = viennals::SpatialSchemeEnum::LAX_FRIEDRICHS_2ND_ORDER;
  VC_TEST_ASSERT(detail::convertSpatialScheme(
                     detail::convertSpatialSchemeToString(scheme)) == scheme);
}

void TestBoundaryConditionConversion() {
  VC_TEST_ASSERT(detail::convertBoundaryCondition(
                     viennals::BoundaryConditionEnum::REFLECTIVE_BOUNDARY) ==
                 viennaray::BoundaryCondition::REFLECTIVE_BOUNDARY);
  VC_TEST_ASSERT(detail::convertBoundaryCondition(
                     viennals::BoundaryConditionEnum::PERIODIC_BOUNDARY) ==
                 viennaray::BoundaryCondition::PERIODIC_BOUNDARY);
}

void TestMetaDataToString() {
  std::unordered_map<std::string, std::vector<double>> metaData;
  metaData["key1"] = {1.0, 2.0};
  std::string str = metaDataToString(metaData);
  std::cout << "Meta data str: " << str << std::endl;
  VC_TEST_ASSERT(str.find("key1") != std::string::npos);
  VC_TEST_ASSERT(str.find("1") != std::string::npos);
}

void TestConvert() {
  // Test convert for SpatialSchemeEnum
  auto scheme = convert<viennals::SpatialSchemeEnum>("EO_1");
  VC_TEST_ASSERT(scheme ==
                 viennals::SpatialSchemeEnum::ENGQUIST_OSHER_1ST_ORDER);

  // Test convert for BoundaryType
  auto boundary = convert<viennaps::BoundaryType>("REFLECTIVE");
  VC_TEST_ASSERT(boundary == viennaps::BoundaryType::REFLECTIVE_BOUNDARY);

  // Test convert for FluxEngineType
  auto fluxEngine = convert<viennaps::FluxEngineType>("CPU_DISK");
  VC_TEST_ASSERT(fluxEngine == viennaps::FluxEngineType::CPU_DISK);

  // Test convert for OxidantType
  auto oxidant = convert<viennaps::OxidantType>("WET");
  VC_TEST_ASSERT(oxidant == viennaps::OxidantType::WET);

  // Test convert for SiliconOrientation
  auto orientation = convert<viennaps::SiliconOrientation>("<110>");
  VC_TEST_ASSERT(orientation == viennaps::SiliconOrientation::Si110);
}

int main() {
  TestSpatialSchemeConversion();
  TestBoundaryConditionConversion();
  TestMetaDataToString();
  std::cout << "Util tests passed!" << std::endl;
  return 0;
}
