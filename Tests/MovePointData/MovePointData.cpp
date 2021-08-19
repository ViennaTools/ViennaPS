#include <cassert>
#include <iostream>
#include <psPointData.hpp>
#include <rayTracingData.hpp>

int main() {
  using NumericType = double;
  psPointData<NumericType> pointData;
  std::vector<NumericType> data(1000);
  pointData.insertNextScalarData(std::move(data), "data");

  rayTracingData<NumericType> rayData;
  rayData.setNumberOfVectorData(pointData.getScalarDataSize());
  rayData.setVectorData(0, std::move(*pointData.getScalarData("data")),
                        pointData.getVectorDataLabel(0));

  assert(pointData.getScalarData("data")->data() == nullptr);

  return 0;
}