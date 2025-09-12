#pragma once

#include <lsMesh.hpp>

#include <vcCudaBuffer.hpp>
#include <vcKDTree.hpp>

#include <array>
#include <vector>

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
    const auto numData = pointData_.getScalarDataSize();
    const auto &elements = surfaceMesh_->triangles;
    const auto numElements = elements.size();
    std::vector<MeshNT> elementData(numData * numElements);
    std::vector<unsigned> dataIdx(numData);

    if (insertToMesh_) {
      std::vector<MeshNT> data(numElements);
      for (unsigned i = 0; i < numData; ++i) {
        const auto label = pointData_.getScalarDataLabel(i);
        surfaceMesh_->getCellData().insertReplaceScalarData(data, label);
        dataIdx[i] = surfaceMesh_->getCellData().getScalarDataIndex(label);
      }
    }

#ifndef NDEBUG
    std::vector<MeshNT> closestPoints(numElements);
#endif

#pragma omp parallel for
    for (unsigned i = 0; i < numElements; ++i) {
      const auto &elIdx = elements[i];
      const auto elementCenter = calculateElementCenter(elIdx);

      const auto closestPoint = pointKdTree_.findNearest(elementCenter);
#ifndef NDEBUG
      closestPoints[i] = closestPoint->first;
#endif

      const auto pointIdx = closestPoint->first;
      for (unsigned j = 0; j < numData; ++j) {
        const auto value = pointData_.getScalarData(j)->at(pointIdx);
        if (upload_) {
          elementData[i + j * numElements] = value;
        }
        if (insertToMesh_) {
          surfaceMesh_->getCellData().getScalarData(dataIdx[j])->at(i) =
              static_cast<MeshNT>(value);
        }
      }
    }

#ifndef NDEBUG
    surfaceMesh_->getCellData().insertReplaceScalarData(closestPoints,
                                                        "pointIds");
    for (unsigned i = 0; i < numData; ++i) {
      std::vector<MeshNT> tmp(elementData.begin() + i * numElements,
                              elementData.begin() + (i + 1) * numElements);
      surfaceMesh_->getCellData().insertReplaceScalarData(
          std::move(tmp), pointData_.getScalarDataLabel(i));
    }
#endif

    if (upload_)
      d_elementData_.allocUpload(elementData);
  }

private:
  // Helper function to calculate triangle center
  std::array<NumericType, 3>
  calculateElementCenter(const std::array<unsigned, 3> &elementIndices) const {
    const auto &nodes = surfaceMesh_->nodes;
    constexpr NumericType oneThird = NumericType(1) / NumericType(3);

    return {(nodes[elementIndices[0]][0] + nodes[elementIndices[1]][0] +
             nodes[elementIndices[2]][0]) *
                oneThird,
            (nodes[elementIndices[0]][1] + nodes[elementIndices[1]][1] +
             nodes[elementIndices[2]][1]) *
                oneThird,
            (nodes[elementIndices[0]][2] + nodes[elementIndices[1]][2] +
             nodes[elementIndices[2]][2]) *
                oneThird};
  }
};

template <class NumericType, class PointNT, class ElemNT, class MeshNT>
class PointToElementDataSingle {
  const std::vector<PointNT> &pointData_;
  std::vector<ElemNT> &elementData_;
  const KDTree<NumericType, Vec3D<NumericType>> &pointKdTree_;
  SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh_;

public:
  PointToElementDataSingle(
      const std::vector<PointNT> &pointData, std::vector<ElemNT> &elementData,
      const KDTree<NumericType, Vec3D<NumericType>> &pointKdTree,
      SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh)
      : pointData_(pointData), elementData_(elementData),
        pointKdTree_(pointKdTree), surfaceMesh_(surfaceMesh) {}

  void apply() {
    const auto &elements = surfaceMesh_->triangles;
    const auto numElements = elements.size();
    elementData_.resize(numElements);

#pragma omp parallel for
    for (unsigned i = 0; i < numElements; ++i) {
      const auto &elIdx = elements[i];
      const auto elementCenter = calculateElementCenter(elIdx);

      const auto closestPoint = pointKdTree_.findNearest(elementCenter);
      const auto value = pointData_.at(closestPoint->first);
      elementData_[i] = static_cast<ElemNT>(value);
    }
  }

private:
  // Helper function to calculate triangle center
  std::array<NumericType, 3>
  calculateElementCenter(const std::array<unsigned, 3> &elementIndices) const {
    const auto &nodes = surfaceMesh_->nodes;
    constexpr NumericType oneThird = NumericType(1) / NumericType(3);

    return {(nodes[elementIndices[0]][0] + nodes[elementIndices[1]][0] +
             nodes[elementIndices[2]][0]) *
                oneThird,
            (nodes[elementIndices[0]][1] + nodes[elementIndices[1]][1] +
             nodes[elementIndices[2]][1]) *
                oneThird,
            (nodes[elementIndices[0]][2] + nodes[elementIndices[1]][2] +
             nodes[elementIndices[2]][2]) *
                oneThird};
  }
};

} // namespace viennaps::gpu
