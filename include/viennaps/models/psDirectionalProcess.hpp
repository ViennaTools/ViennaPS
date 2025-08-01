#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <lsCalculateVisibilities.hpp>
#include <vcVectorType.hpp>

#include <vector>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <class NumericType> struct RateSet {
  Vec3D<NumericType> direction;
  NumericType directionalVelocity;
  NumericType isotropicVelocity;
  std::vector<Material> maskMaterials;
  bool calculateVisibility;

  RateSet(const Vec3D<NumericType> &dir = Vec3D<NumericType>{0., 0., 0.},
          NumericType dirVel = 0., NumericType isoVel = 0.,
          const std::vector<Material> &masks =
              std::vector<Material>{
                  Material::Mask}, // Default to Material::Mask
          bool calcVis = true)
      : direction(dir), directionalVelocity(dirVel), isotropicVelocity(isoVel),
        maskMaterials(masks),
        calculateVisibility(calcVis &&
                            (dir[0] != 0. || dir[1] != 0. || dir[2] != 0.) &&
                            dirVel != 0) {}

  void print() const {
    std::cout << "RateSet: " << std::endl;
    std::cout << "Direction: " << direction << std::endl;
    std::cout << "Directional Velocity: " << directionalVelocity << std::endl;
    std::cout << "Isotropic Velocity: " << isotropicVelocity << std::endl;
    std::cout << "Mask Materials: ";
    for (const auto &mask : maskMaterials) {
      std::cout << MaterialMap::getMaterialName(mask) << " ";
    }
    std::cout << std::endl;
    std::cout << "Calculate Visibility: " << calculateVisibility << std::endl;
  }
};

template <class NumericType, int D>
class DirectionalVelocityField : public VelocityField<NumericType, D> {
  const std::vector<RateSet<NumericType>> rateSets_;
  std::unordered_map<unsigned, std::vector<NumericType>> visibilities_;

public:
  DirectionalVelocityField(std::vector<RateSet<NumericType>> &&rateSets)
      : rateSets_(std::move(rateSets)) {}

  NumericType getScalarVelocity(const Vec3D<NumericType> &coordinate,
                                int material,
                                const Vec3D<NumericType> &normalVector,
                                unsigned long pointId) override {
    NumericType scalarVelocity = 0.0;

    for (const auto &rateSet : rateSets_) {
      if (isMaskMaterial(material, rateSet.maskMaterials)) {
        continue; // Skip this rate set if material is masked
      }
      // Accumulate isotropic velocities
      scalarVelocity += rateSet.isotropicVelocity;
    }

    return scalarVelocity;
  }

  Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType> &coordinate,
                                       int material,
                                       const Vec3D<NumericType> &normalVector,
                                       unsigned long pointId) override {
    Vec3D<NumericType> vectorVelocity{0., 0., 0.};

    for (unsigned rateSetID = 0; rateSetID < rateSets_.size(); ++rateSetID) {
      const auto &rateSet = rateSets_[rateSetID];
      if (isMaskMaterial(material, rateSet.maskMaterials)) {
        continue; // Skip this rate set if material is masked
      }

      if (rateSet.calculateVisibility &&
          visibilities_[rateSetID].at(pointId) == 0.) {
        continue; // Skip if visibility check fails
      }

      // Calculate the potential velocity vector for this rate set
      Vec3D<NumericType> potentialVelocity =
          rateSet.direction * rateSet.directionalVelocity;

      NumericType dotProduct = DotProduct(potentialVelocity, normalVector);
      if (dotProduct != 0) {
        vectorVelocity = vectorVelocity - potentialVelocity;
      }
    }

    return vectorVelocity;
  }

  // The translation field should be disabled when using a surface model
  // which only depends on an analytic velocity field
  int getTranslationFieldOptions() const override { return 0; }

  void prepare(SmartPointer<Domain<NumericType, D>> domain,
               SmartPointer<std::vector<NumericType>> velocities,
               const NumericType processTime) override {

    visibilities_.clear();

    // Calculate visibilities for each rate set
    auto surfaceLS = domain->getLevelSets().back();
    for (unsigned rateSetID = 0; rateSetID < rateSets_.size(); ++rateSetID) {
      auto &rateSet = rateSets_[rateSetID];
      if (rateSet.calculateVisibility) {

        std::string label = "Visibilities_" + std::to_string(rateSetID);
        viennals::CalculateVisibilities<NumericType, D>(
            surfaceLS, rateSet.direction, label)
            .apply();
        visibilities_[rateSetID] =
            *surfaceLS->getPointData().getScalarData(label); // Copy
      }
    }
  }

  const std::vector<RateSet<NumericType>> &getRateSets() const {
    return rateSets_;
  }
  const std::unordered_map<unsigned, std::vector<NumericType>> &
  getVisibilities() const {
    return visibilities_;
  }

protected:
  static bool isMaskMaterial(const int material,
                             const std::vector<Material> &maskMaterials) {
    for (const auto &maskMaterial : maskMaterials) {
      if (MaterialMap::isMaterial(material, maskMaterial)) {
        return true;
      }
    }
    return false;
  }
};

} // namespace impl

/// Directional rate with multiple rate sets and masking materials.
template <typename NumericType, int D>
class DirectionalProcess : public ProcessModel<NumericType, D> {
public:
  using RateSet = impl::RateSet<NumericType>;

  DirectionalProcess(const Vec3D<NumericType> &direction,
                     NumericType directionalVelocity,
                     NumericType isotropicVelocity = 0.,
                     const Material maskMaterial = Material::Mask,
                     bool calculateVisibility = true) {
    std::vector<RateSet> rateSets;
    rateSets.emplace_back(direction, directionalVelocity, isotropicVelocity,
                          std::vector<Material>{maskMaterial},
                          calculateVisibility);
    initialize(std::move(rateSets));
  }

  // Constructor accepting direction, directional velocity, isotropic velocity,
  // and optional mask materials
  DirectionalProcess(const Vec3D<NumericType> &direction,
                     NumericType directionalVelocity,
                     NumericType isotropicVelocity,
                     const std::vector<Material> &maskMaterials,
                     bool calculateVisibility = true) {
    std::vector<RateSet> rateSets;
    rateSets.emplace_back(direction, directionalVelocity, isotropicVelocity,
                          maskMaterials, calculateVisibility);
    initialize(std::move(rateSets));
  }

  // Constructor accepting single rate set
  DirectionalProcess(const RateSet &rateSet) {
    std::vector<RateSet> rateSets;
    rateSets.push_back(rateSet);
    initialize(std::move(rateSets));
  }

  // Constructor accepting multiple rate sets
  DirectionalProcess(std::vector<RateSet> rateSets) {
    initialize(std::move(rateSets));
  }

private:
  void initialize(std::vector<RateSet> &&rateSets) {
    // Default surface model
    auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

    // Velocity field with multiple rate sets
    auto velField =
        SmartPointer<impl::DirectionalVelocityField<NumericType, D>>::New(
            std::move(rateSets));

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("DirectionalProcess");

    // Store process data
    processMetaData["DirectionalVelocity"] = std::vector<NumericType>();
    processMetaData["IsotropicVelocity"] = std::vector<NumericType>();
    processMetaData["MaskMaterials"] = std::vector<NumericType>();
    processMetaData["CalculateVisibility"] = std::vector<NumericType>();
    int i = 0;
    for (const auto &rateSet : rateSets) {
      processMetaData["DirectionalVelocity"].push_back(
          rateSet.directionalVelocity);
      processMetaData["IsotropicVelocity"].push_back(rateSet.isotropicVelocity);
      for (const auto &maskMaterial : rateSet.maskMaterials) {
        processMetaData["MaskMaterials"].push_back(
            static_cast<NumericType>(maskMaterial));
      }
      processMetaData["CalculateVisibility"].push_back(
          static_cast<NumericType>(rateSet.calculateVisibility ? 1 : 0));
      processMetaData["Direction " + std::to_string(i++)] =
          std::vector<NumericType>{rateSet.direction[0], rateSet.direction[1],
                                   rateSet.direction[2]};
    }
  }

  using ProcessModel<NumericType, D>::processMetaData;
};

PS_PRECOMPILE_PRECISION_DIMENSION(DirectionalProcess)

} // namespace viennaps
