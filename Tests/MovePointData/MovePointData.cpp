#include <psPointData.hpp>
#include <rayTracingData.hpp>
#include <iostream>
#include <cassert>

int main()
{
    using NumericType = double;
    psPointData<NumericType> pointData;
    std::vector<NumericType> data(1000);
    pointData.insertNextScalarData(std::move(data), "data");

    rayTracingData<NumericType> rayData;
    rayData.setNumberOfVectorData(pointData.getScalarDataSize());
    rayData.getVectorData(0) = std::move(*pointData.getScalarData("data"));

    assert(pointData.getScalarData("data")->data() == nullptr);

    return 0;
}