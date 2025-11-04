#pragma once

#include <lsMesh.hpp>

#include <vcCudaBuffer.hpp>
#include <vcKDTree.hpp>
#include <vcVectorType.hpp>

#include <array>
#include <vector>

namespace viennaps {

using namespace viennacore;

template <class NumericType, class MeshNT = NumericType>
class PointToElementDataBase {
protected:
  viennals::PointData<NumericType> &pointData_;
  KDTree<NumericType, Vec3D<NumericType>> const &pointKdTree_;
  SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh_;

  const bool insertToMesh_ = false;

public:
  PointToElementDataBase(
      viennals::PointData<NumericType> &pointData,
      KDTree<NumericType, Vec3D<NumericType>> const &pointKdTree,
      SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh,
      bool insertToMesh = false)
      : pointData_(pointData), pointKdTree_(pointKdTree),
        surfaceMesh_(surfaceMesh), insertToMesh_(insertToMesh) {}

  void apply() {
    const auto numData = pointData_.getScalarDataSize();
    const auto &elements = surfaceMesh_->triangles;
    const auto numElements = elements.size();
    std::vector<unsigned> dataIdx(numData);

    prepareData();

    if (insertToMesh_) {
      for (unsigned i = 0; i < numData; ++i) {
        std::vector<MeshNT> data(numElements);
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
        setDataAtElement(i, j, value);
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

    finalizeData();
  }

protected:
  virtual void prepareData() {}

  virtual void finalizeData() {}

  virtual void setDataAtElement(unsigned elementIdx, unsigned dataIdx,
                                NumericType value) = 0;

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

template <class NumericType, class MeshNT = NumericType>
class PointToElementData : public PointToElementDataBase<NumericType, MeshNT> {

public:
  PointToElementData(viennals::PointData<NumericType> &elementData,
                     SmartPointer<viennals::PointData<NumericType>> pointData,
                     KDTree<NumericType, Vec3D<NumericType>> const &pointKdTree,
                     SmartPointer<viennals::Mesh<MeshNT>> &surfaceMesh,
                     bool insertToMesh = false)
      : PointToElementDataBase<NumericType, MeshNT>(*pointData, pointKdTree,
                                                    surfaceMesh, insertToMesh),
        elementData_(elementData) {}

  PointToElementData(viennals::PointData<NumericType> &elementData,
                     viennals::PointData<NumericType> &pointData,
                     KDTree<NumericType, Vec3D<NumericType>> &pointKdTree,
                     SmartPointer<viennals::Mesh<MeshNT>> &surfaceMesh,
                     bool insertToMesh = false)
      : PointToElementDataBase<NumericType, MeshNT>(pointData, pointKdTree,
                                                    surfaceMesh, insertToMesh),
        elementData_(elementData) {}

protected:
  void prepareData() override {
    const auto numData = this->pointData_.getScalarDataSize();
    const auto &elements = this->surfaceMesh_->triangles;
    const auto numElements = elements.size();

    dataIdx.resize(numData);
    for (unsigned i = 0; i < numData; ++i) {
      std::vector<NumericType> data(numElements);
      const auto label = this->pointData_.getScalarDataLabel(i);
      elementData_.insertReplaceScalarData(data, label);
      dataIdx[i] = elementData_.getScalarDataIndex(label);
    }
  }

  void setDataAtElement(unsigned elementIdx, unsigned dataIdxGlobal,
                        NumericType value) override {
    elementData_.getScalarData(dataIdx[dataIdxGlobal])->at(elementIdx) = value;
  }

private:
  std::vector<unsigned> dataIdx;
  viennals::PointData<NumericType> &elementData_;
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

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {
// Same as PointToElementData but stores data in a CudaBuffer
template <class NumericType, class MeshNT = NumericType>
class PointToElementData
    : public ::viennaps::PointToElementDataBase<NumericType, MeshNT> {

  CudaBuffer &d_elementData_;
  std::vector<MeshNT> elementData_;
  unsigned numElements_ = 0;

public:
  PointToElementData(CudaBuffer &d_elementData,
                     SmartPointer<viennals::PointData<NumericType>> pointData,
                     KDTree<NumericType, Vec3D<NumericType>> const &pointKdTree,
                     SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh,
                     bool insertToMesh = false)
      : ::viennaps::PointToElementDataBase<NumericType, MeshNT>(
            *pointData, pointKdTree, surfaceMesh, insertToMesh),
        d_elementData_(d_elementData) {}

  PointToElementData(CudaBuffer &d_elementData,
                     viennals::PointData<NumericType> &pointData,
                     KDTree<NumericType, Vec3D<NumericType>> &pointKdTree,
                     SmartPointer<viennals::Mesh<MeshNT>> surfaceMesh,
                     bool insertToMesh = false)
      : ::viennaps::PointToElementDataBase<NumericType, MeshNT>(
            pointData, pointKdTree, surfaceMesh, insertToMesh),
        d_elementData_(d_elementData) {}

protected:
  void prepareData() override {
    const auto numData = this->pointData_.getScalarDataSize();
    const auto &elements = this->surfaceMesh_->triangles;
    numElements_ = elements.size();
    elementData_.resize(numData * numElements_);
  }

  void finalizeData() override { d_elementData_.allocUpload(elementData_); }

  void setDataAtElement(unsigned elementIdx, unsigned dataIdx,
                        NumericType value) override {
    elementData_[elementIdx + dataIdx * numElements_] = value;
  }
};
} // namespace gpu
#endif

} // namespace viennaps
