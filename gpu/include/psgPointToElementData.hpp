#pragma once

#include <lsMesh.hpp>

#include <vcCudaBuffer.hpp>
#include <vcKDTree.hpp>

namespace viennaps::gpu {

template <class NumericType, class MeshNT = NumericType>
class PointToElementData {

  viennals::PointData<NumericType> &pointData_;
  KDTree<NumericType, Vec3D<NumericType>> const &pointKdTree_;
  SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh_;
  CudaBuffer &d_elementData_;

  const bool insertToMesh_ = false;
  const bool upload_ = true;

public:
  PointToElementData(CudaBuffer &d_elementData,
                     SmartPointer<viennals::PointData<NumericType>> pointData,
                     KDTree<NumericType, Vec3D<NumericType>> const &pointKdTree,
                     SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh,
                     bool insertToMesh = false, bool upload = true)
      : pointData_(*pointData), pointKdTree_(pointKdTree),
        surfaceMesh_(surfaceMesh), d_elementData_(d_elementData),
        insertToMesh_(insertToMesh), upload_(upload) {}

  PointToElementData(CudaBuffer &d_elementData,
                     viennals::PointData<NumericType> &pointData,
                     KDTree<NumericType, Vec3D<NumericType>> &pointKdTree,
                     SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh,
                     bool insertToMesh = false, bool upload = true)
      : pointData_(pointData), pointKdTree_(pointKdTree),
        surfaceMesh_(surfaceMesh), d_elementData_(d_elementData),
        insertToMesh_(insertToMesh), upload_(upload) {}

  void apply() {

    auto numData = pointData_.getScalarDataSize();
    const auto &elements = surfaceMesh_->triangles;
    auto numElements = elements.size();
    std::vector<MeshNT> elementData(numData * numElements);
    std::vector<unsigned> dataIdx(numData);

    if (insertToMesh_) {
      std::vector<MeshNT> data(numElements);
      for (unsigned i = 0; i < numData; i++) {
        auto label = pointData_.getScalarDataLabel(i);
        surfaceMesh_->getCellData().insertReplaceScalarData(data, label);
        dataIdx[i] = surfaceMesh_->getCellData().getScalarDataIndex(label);
      }
    }

#ifndef NDEBUG
    std::vector<MeshNT> closestPoints(numElements);
#endif

#pragma omp parallel for
    for (unsigned i = 0; i < numElements; i++) {
      auto &elIdx = elements[i];
      std::array<NumericType, 3> elementCenter{
          static_cast<NumericType>((surfaceMesh_->nodes[elIdx[0]][0] +
                                    surfaceMesh_->nodes[elIdx[1]][0] +
                                    surfaceMesh_->nodes[elIdx[2]][0]) /
                                   3.f),
          static_cast<NumericType>((surfaceMesh_->nodes[elIdx[0]][1] +
                                    surfaceMesh_->nodes[elIdx[1]][1] +
                                    surfaceMesh_->nodes[elIdx[2]][1]) /
                                   3.f),
          static_cast<NumericType>((surfaceMesh_->nodes[elIdx[0]][2] +
                                    surfaceMesh_->nodes[elIdx[1]][2] +
                                    surfaceMesh_->nodes[elIdx[2]][2]) /
                                   3.f)};

      auto closestPoint = pointKdTree_.findNearest(elementCenter);
#ifndef NDEBUG
      closestPoints[i] = closestPoint->first;
#endif

      for (unsigned j = 0; j < numData; j++) {
        const auto value = pointData_.getScalarData(j)->at(closestPoint->first);
        if (upload_)
          elementData[i + j * numElements] = value;
        if (insertToMesh_)
          surfaceMesh_->getCellData().getScalarData(dataIdx[j])->at(i) =
              static_cast<MeshNT>(value);
      }
    }

#ifndef NDEBUG
    surfaceMesh_->getCellData().insertReplaceScalarData(closestPoints,
                                                        "pointIds");
    for (int i = 0; i < numData; i++) {
      std::vector<MeshNT> tmp(elementData.begin() + i * numElements,
                              elementData.begin() + (i + 1) * numElements);
      surfaceMesh_->getCellData().insertReplaceScalarData(
          std::move(tmp), pointData_.getScalarDataLabel(i));
    }
#endif

    if (upload_)
      d_elementData_.allocUpload(elementData);
  }
};

} // namespace viennaps::gpu
