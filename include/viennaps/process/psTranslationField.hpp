#pragma once

#include "../materials/psMaterialMap.hpp"
#include "psVelocityField.hpp"

#include <lsVelocityField.hpp>

#include <vcKDTree.hpp>
#include <vcLogger.hpp>
#include <vcNFKDTree.hpp>
#include <vcSmartPointer.hpp>
#include <vcVectorType.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <utility>
#include <vector>

namespace viennaps {

using namespace viennacore;

template <Numeric NumericType, int D,
          class TreeType = NFKDTree<NumericType, Vec3D<NumericType>>>
  requires Dimension<D>
class TranslationField final : public viennals::VelocityField<NumericType> {
public:
  using TranslatorType = std::unordered_map<unsigned long, unsigned long>;
  using KDTreeType = TreeType;

public:
  TranslationField(
      SmartPointer<::viennaps::VelocityField<NumericType, D>> velocityField,
      SmartPointer<MaterialMap> const &materialMap, int translationMethod)
      : modelVelocityField_(velocityField), materialMap_(materialMap),
        translationMethod_(translationMethod) {
    if (!materialMap_) {
      VIENNACORE_LOG_ERROR("TranslationField: material map is required.");
    }
  }

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                int material,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {
    material = materialMap_->getMaterialIdAtIdx(material);
    assert(material >= 0);
    NumericType velocity;
    if (translationMethod_ == 2 &&
        facingVelocity(coordinate, material, normalVector, velocity))
      return velocity;
    translateLsId(pointId, coordinate, material);
    return modelVelocityField_->getScalarVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType> &coordinate,
                                       int material,
                                       const Vec3D<NumericType> &normalVector,
                                       unsigned long pointId) override {
    material = materialMap_->getMaterialIdAtIdx(material);
    assert(material >= 0);
    translateLsId(pointId, coordinate, material);
    return modelVelocityField_->getVectorVelocity(coordinate, material,
                                                  normalVector, pointId);
  }

  NumericType
  getDissipationAlpha(int direction, int material,
                      const Vec3D<NumericType> &centralDifferences) override {
    material = materialMap_->getMaterialIdAtIdx(material);
    assert(material >= 0);
    return modelVelocityField_->getDissipationAlpha(direction, material,
                                                    centralDifferences);
  }

  void setTranslator(const SmartPointer<TranslatorType> &translator) {
    translator_ = translator;
  }

  void setKdTree(const SmartPointer<KDTreeType> kdTree) { kdTree_ = kdTree; }

  auto &getKdTree() { return kdTree_; }

  /// Trees of the disks of every material, so that a point takes the
  /// velocity of the nearest disk of the material asked for. Velocity fields
  /// keep velocities per disk, computed with the material of the disk, and a
  /// Euclidean surface asks for the velocities of the materials on either
  /// side of an interface where it ends, as at the edge of a mask. Given the
  /// normals of the disks and the grid spacing, a scalar velocity asked for
  /// with a normal comes from the disks facing that way (facingVelocity).
  void
  buildMaterialKdTrees(const std::vector<Vec3D<NumericType>> &points,
                       const std::vector<NumericType> &materialIds,
                       const std::vector<Vec3D<NumericType>> *normals = nullptr,
                       NumericType gridDelta = 0) {
    diskNormals_ = normals ? *normals : std::vector<Vec3D<NumericType>>{};
    diskPoints_ = points;
    gridDelta_ = gridDelta;
    crossTrees_.clear();
    crossDisks_.clear();
    crossVelocities_.clear();
    materialTrees_.clear();
    std::unordered_map<int, std::vector<Vec3D<NumericType>>> grouped;
    for (std::size_t k = 0; k < points.size() && k < materialIds.size(); ++k) {
      const int id = static_cast<int>(materialIds[k]);
      grouped[id].push_back(points[k]);
      materialTrees_[id].second.push_back(k);
    }
    for (auto &[id, group] : grouped) {
      auto &tree = materialTrees_[id].first;
      tree = SmartPointer<KDTreeType>::New();
      tree->setPoints(group);
      tree->build();
    }
  }

  /// Velocities of the disks at a junction of materials for a material
  /// across the junction, computed from each disk's own fluxes (entry k for
  /// disk disks[k] and material materials[k]). A front of one material
  /// facing a direction that only a disk of another material faces, as below
  /// the edge of a mask where the undercut faces the same way as the mask
  /// wall, takes the velocity its own material has under the fluxes that
  /// disk receives.
  void setCrossVelocities(const std::vector<Vec3D<NumericType>> &points,
                          const std::vector<unsigned long> &disks,
                          const std::vector<NumericType> &materials,
                          const std::vector<NumericType> &velocities) {
    crossTrees_.clear();
    crossDisks_ = disks;
    crossVelocities_ = velocities;
    std::unordered_map<int, std::vector<Vec3D<NumericType>>> grouped;
    for (std::size_t k = 0;
         k < disks.size() && k < materials.size() && k < velocities.size();
         ++k) {
      const int id = static_cast<int>(materials[k]);
      grouped[id].push_back(points[disks[k]]);
      crossTrees_[id].second.push_back(k);
    }
    for (auto &[id, group] : grouped) {
      auto &tree = crossTrees_[id].first;
      tree = SmartPointer<KDTreeType>::New();
      tree->setPoints(group);
      tree->build();
    }
  }

  void buildKdTree(const std::vector<std::array<NumericType, 3>> &points) {
    if (!kdTree_)
      kdTree_ = SmartPointer<KDTreeType>::New();
    kdTree_->setPoints(points);
    kdTree_->build();
  }

  void translateLsId(unsigned long &lsId, const Vec3D<NumericType> &coordinate,
                     int materialId = -1) const {
    switch (translationMethod_) {
    case 1: {
      assert(translator_->size() > 0);
      if (auto it = translator_->find(lsId); it != translator_->end()) {
        lsId = it->second;
      } else {
        Logger::getInstance()
            .addError("Could not extend velocity from surface (" +
                          std::to_string(lsId) + ") to LS point",
                      false)
            .print();
      }
      break;
    }
    case 2: {
      if (const long disk = findNearestDisk(coordinate, materialId); disk >= 0)
        lsId = static_cast<unsigned long>(disk);
      break;
    }
    default:
      break;
    }
  }

  /// Index of the disk nearest to a point among the disks of the material
  /// asked for, or among all disks where that material has none; -1 without
  /// disks.
  long findNearestDisk(const Vec3D<NumericType> &coordinate,
                       int materialId = -1) const {
    if (auto found = materialTrees_.find(materialId);
        found != materialTrees_.end())
      if (auto nearest = found->second.first->findNearest(coordinate))
        return static_cast<long>(found->second.second[nearest->first]);
    if (kdTree_)
      if (auto nearest = kdTree_->findNearest(coordinate))
        return static_cast<long>(nearest->first);
    return -1;
  }

  /// The scalar velocity of a point asked for with a normal, from the disks
  /// within 1.5 grid spacings facing its half space: disks of its material,
  /// and disks of other materials at a junction with their velocities for
  /// that material, one for each direction (the nearest within 10 degrees,
  /// a disk of the material first). A disk facing the normal within a degree
  /// gives its velocity. Else two disks whose normals enclose the normal give
  /// the velocity interpolated linearly in angle between theirs, as for a
  /// direction of a fan between the faces meeting at an edge, else the disk
  /// facing it within 45 degrees. A fan asks for many directions at its
  /// feature point, where the nearest disk lies on one of the faces meeting
  /// there: at the edge of a mask the floor's rate would drive the undercut
  /// in every direction. Interpolated velocities stay below those that keep
  /// an edge sharp where its faces move at different speeds and give the arc
  /// of a uniform speed. False where no disk faces the half space.
  bool facingVelocity(const Vec3D<NumericType> &coordinate, int materialId,
                      const Vec3D<NumericType> &normal, NumericType &velocity) {
    if (gridDelta_ <= 0 || diskNormals_.empty())
      return false;
    const NumericType radius2 = NumericType(2.25) * gridDelta_ * gridDelta_;
    struct Candidate {
      long disk;   // index among the disks, -1 for a velocity across
      long across; // index among the velocities across a junction
      Vec3D<NumericType> normal;
      NumericType angle;
    };
    std::vector<Candidate> chosen;
    auto gather = [&](const KDTreeType &tree, auto diskOf, auto acrossOf) {
      const auto found = tree.findKNearest(coordinate, 16);
      if (!found)
        return;
      for (const auto &[k, distance2] : *found) {
        if (distance2 > radius2)
          break;
        const auto &n = diskNormals_[diskOf(k)];
        const NumericType c = DotProduct(n, normal);
        if (c <= 0)
          continue;
        // a disk of the surface at the point, whose tangent plane passes the
        // point within a grid spacing, as the faces meeting at an edge there:
        // a floor a little below the foot of a wall faces the tilted
        // directions of the wall as well, but its speed is not theirs
        if (diskOf(k) < diskPoints_.size()) {
          const auto &q = diskPoints_[diskOf(k)];
          NumericType offset = 0;
          for (int i = 0; i < 3; ++i)
            offset += n[i] * (coordinate[i] - q[i]);
          if (std::abs(offset) > gridDelta_)
            continue;
        }
        bool seen = false;
        for (const auto &other : chosen)
          seen = seen || DotProduct(n, other.normal) >
                             NumericType(0.98480775301220806); // 10 degrees
        if (!seen)
          chosen.push_back({acrossOf(k) < 0 ? long(diskOf(k)) : -1, acrossOf(k),
                            n, std::acos(std::min(c, NumericType(1)))});
      }
    };
    if (auto found = materialTrees_.find(materialId);
        found != materialTrees_.end()) {
      const auto &indices = found->second.second;
      gather(
          *found->second.first, [&](std::size_t k) { return indices[k]; },
          [](std::size_t) { return long(-1); });
    }
    if (auto found = crossTrees_.find(materialId); found != crossTrees_.end()) {
      const auto &entries = found->second.second;
      gather(
          *found->second.first,
          [&](std::size_t k) { return crossDisks_[entries[k]]; },
          [&](std::size_t k) { return long(entries[k]); });
    }
    if (chosen.empty())
      return false;
    auto velocityOf = [&](const Candidate &c) {
      if (c.across >= 0)
        return crossVelocities_[c.across];
      return modelVelocityField_->getScalarVelocity(
          coordinate, materialId, normal, static_cast<unsigned long>(c.disk));
    };
    const NumericType degree = NumericType(0.017453292519943295);
    std::size_t best = 0;
    for (std::size_t k = 1; k < chosen.size(); ++k)
      if (chosen[k].angle < chosen[best].angle)
        best = k;
    if (chosen[best].angle < degree) {
      velocity = velocityOf(chosen[best]);
      return true;
    }
    long first = -1, second = -1;
    NumericType span = 0;
    for (std::size_t a = 0; a < chosen.size(); ++a)
      for (std::size_t b = a + 1; b < chosen.size(); ++b) {
        const NumericType between =
            std::acos(std::clamp(DotProduct(chosen[a].normal, chosen[b].normal),
                                 NumericType(-1), NumericType(1)));
        const NumericType sum = chosen[a].angle + chosen[b].angle;
        if (std::abs(sum - between) < degree && (first < 0 || sum < span)) {
          first = long(a);
          second = long(b);
          span = sum;
        }
      }
    if (first >= 0) {
      const auto &a = chosen[first], &b = chosen[second];
      velocity = (velocityOf(a) * b.angle + velocityOf(b) * a.angle) /
                 (a.angle + b.angle);
      return true;
    }
    if (chosen[best].angle <= NumericType(0.78539816339744831)) { // 45 degrees
      velocity = velocityOf(chosen[best]);
      return true;
    }
    return false;
  }

  auto const getTranslationMethod() const { return translationMethod_; }

private:
  std::vector<Vec3D<NumericType>> diskNormals_, diskPoints_;
  NumericType gridDelta_ = 0;
  // velocities of disks at junctions for the materials across them: their
  // trees per material, the disks and the velocities
  std::unordered_map<
      int, std::pair<SmartPointer<KDTreeType>, std::vector<unsigned long>>>
      crossTrees_;
  std::vector<unsigned long> crossDisks_;
  std::vector<NumericType> crossVelocities_;
  SmartPointer<TranslatorType> translator_;
  SmartPointer<KDTreeType> kdTree_;
  // disks of every material: their tree and their indices among all disks
  std::unordered_map<
      int, std::pair<SmartPointer<KDTreeType>, std::vector<unsigned long>>>
      materialTrees_;

  const SmartPointer<::viennaps::VelocityField<NumericType, D>>
      modelVelocityField_;
  const SmartPointer<MaterialMap> materialMap_;
  const int translationMethod_ = 1;
};

} // namespace viennaps
