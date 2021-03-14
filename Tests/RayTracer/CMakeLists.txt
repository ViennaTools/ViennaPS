cmake_minimum_required(VERSION 3.4)

project("RayTracer")

find_package(ViennaHRLE REQUIRED)
find_package(ViennaLS REQUIRED)
find_package (rtidevice REQUIRED)


add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC "${VIENNAPS_INCLUDE_DIRS}")
target_link_libraries(${PROJECT_NAME} PRIVATE ${VIENNAPS_LIBRARIES})
target_link_libraries (${PROJECT_NAME} PRIVATE rtidevice::rtidevice)
