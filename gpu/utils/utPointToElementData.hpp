#pragma once

#include <vcKDTree.hpp>

namespace viennaps {

namespace gpu {

template <class NumericType> class PointToElementData {

  viennals::PointData<NumericType> &pointData_;
  SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> pointKdTree_;
  SmartPointer<viennals::Mesh<float>> surfaceMesh_;
  CudaBuffer &d_elementData_;

  const bool insertToMesh_ = false;
  const bool upload_ = true;

public:
  PointToElementData(
      CudaBuffer &d_elementData,
      SmartPointer<viennals::PointData<NumericType>> pointData,
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> pointKdTree,
      SmartPointer<viennals::Mesh<float>> surfaceMesh,
      bool insertToMesh = false, bool upload = true)
      : d_elementData_(d_elementData), pointData_(*pointData),
        pointKdTree_(pointKdTree), surfaceMesh_(surfaceMesh),
        insertToMesh_(insertToMesh), upload_(upload) {}

  PointToElementData(
      CudaBuffer &d_elementData, viennals::PointData<NumericType> &pointData,
      SmartPointer<KDTree<NumericType, Vec3D<NumericType>>> pointKdTree,
      SmartPointer<viennals::Mesh<float>> surfaceMesh,
      bool insertToMesh = false, bool upload = true)
      : d_elementData_(d_elementData), pointData_(pointData),
        pointKdTree_(pointKdTree), surfaceMesh_(surfaceMesh),
        insertToMesh_(insertToMesh), upload_(upload) {}

  void apply() {

    auto numData = pointData_.getScalarDataSize();
    const auto &elements = surfaceMesh_->triangles;
    auto numElements = elements.size();
    std::vector<NumericType> elementData(numData * numElements);
    std::vector<unsigned> dataIdx(numData);

    if (insertToMesh_) {
      std::vector<NumericType> data(numElements);
      for (unsigned i = 0; i < numData; i++) {
        auto label = pointData_.getScalarDataLabel(i);
        surfaceMesh_->getCellData().insertReplaceScalarData(data, label);
        dataIdx[i] = surfaceMesh_->getCellData().getScalarDataIndex(label);
      }
    }

#ifndef NDEBUG
    std::vector<NumericType> closestPoints(numElements);
#endif

#pragma omp parallel for
    for (unsigned i = 0; i < numElements; i++) {
      auto &elIdx = elements[i];
      std::array<NumericType, 3> elementCenter{
          (surfaceMesh_->nodes[elIdx[0]][0] + surfaceMesh_->nodes[elIdx[1]][0] +
           surfaceMesh_->nodes[elIdx[2]][0]) /
              3.f,
          (surfaceMesh_->nodes[elIdx[0]][1] + surfaceMesh_->nodes[elIdx[1]][1] +
           surfaceMesh_->nodes[elIdx[2]][1]) /
              3.f,
          (surfaceMesh_->nodes[elIdx[0]][2] + surfaceMesh_->nodes[elIdx[1]][2] +
           surfaceMesh_->nodes[elIdx[2]][2]) /
              3.f};

      auto closestPoint = pointKdTree_->findNearest(elementCenter);
#ifndef NDEBUG
      closestPoints[i] = closestPoint->first;
#endif

      for (unsigned j = 0; j < numData; j++) {
        const auto value = pointData_.getScalarData(j)->at(closestPoint->first);
        if (upload_)
          elementData[i + j * numElements] = value;
        if (insertToMesh_)
          surfaceMesh_->getCellData().getScalarData(dataIdx[j])->at(i) = value;
      }
    }

#ifndef NDEBUG
    surfaceMesh_->getCellData().insertReplaceScalarData(closestPoints,
                                                        "pointIds");
    for (int i = 0; i < numData; i++) {
      std::vector<NumericType> tmp(elementData.begin() + i * numElements,
                                   elementData.begin() + (i + 1) * numElements);
      surfaceMesh_->getCellData().insertReplaceScalarData(
          std::move(tmp), pointData_.getScalarDataLabel(i));
    }
#endif

    if (upload_)
      d_elementData_.allocUpload(elementData);
  }
};

} // namespace gpu
} // namespace viennaps