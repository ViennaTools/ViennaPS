#pragma once

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <vcVectorUtil.hpp>
#include <vector>
#include <algorithm>

namespace viennaps {

using namespace viennacore;

namespace impl {

template <class NumericType, int D>
struct RateSet {
    Vec3D<NumericType> direction;
    NumericType directionalVelocity;
    NumericType isotropicVelocity;
    std::vector<int> maskMaterials;
    bool calculateVisibility;

    RateSet(const Vec3D<NumericType>& dir = Vec3D<NumericType>{0., 0., 0.},
            NumericType dirVel = 0.,
            NumericType isoVel = 0.,
            const std::vector<int>& masks = std::vector<int>{Material::Mask}, // Default to Material::Mask
            // const std::vector<int>& masks = {},
            bool calcVis = true)
        : direction(dir), directionalVelocity(dirVel),
          isotropicVelocity(isoVel), maskMaterials(masks),
          calculateVisibility(calcVis && (dir[0] != 0. || dir[1] != 0. || dir[2] != 0.) && dirVel != 0) { }
};

template <class NumericType, int D>
class DirectionalEtchVelocityField : public VelocityField<NumericType> {
    const std::vector<RateSet<NumericType, D>> rateSets_;
    const bool useVisibilities_;

public:
    DirectionalEtchVelocityField(const std::vector<RateSet<NumericType, D>>& rateSets,
                                 const bool useVisibilities)
        : rateSets_(rateSets), useVisibilities_(useVisibilities) {}

    NumericType getScalarVelocity(const Vec3D<NumericType>& coordinate,
                                  int material,
                                  const Vec3D<NumericType>& normalVector,
                                  unsigned long pointId) override {
        NumericType scalarVelocity = 0.0;

        for (const auto& rateSet : rateSets_) {
            if (isMaskMaterial(material, rateSet.maskMaterials)) {
                continue; // Skip this rate set if material is masked
            }
            // Accumulate isotropic velocities
            scalarVelocity += std::min(0., rateSet.isotropicVelocity);
        }

        return scalarVelocity;
    }

    Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType>& coordinate,
                                         int material,
                                         const Vec3D<NumericType>& normalVector,
                                         unsigned long pointId) override {
        Vec3D<NumericType> vectorVelocity = {0.0, 0.0, 0.0};

        // for (const auto& rateSet : rateSets_) {
        for (int rateSetID = 0; rateSetID < rateSets_.size(); ++rateSetID) {
            const auto& rateSet = rateSets_[rateSetID];
            if (isMaskMaterial(material, rateSet.maskMaterials)) {
                continue; // Skip this rate set if material is masked
            }
            // if (useVisibilities_ && this->visibilities_[rateSetID]->at(pointId) == 0.) {
            if (rateSet.calculateVisibility && this->visibilities_[rateSetID]->at(pointId) == 0.) {
                continue; // Skip if visibility check fails
            }

            // Calculate the potential velocity vector for this rate set
            Vec3D<NumericType> potentialVelocity = rateSet.direction * rateSet.directionalVelocity;

            // Compute dot product manually for std::array
            NumericType dotProduct = 0.0;
            for (int i = 0; i < 3; ++i) {
                dotProduct += potentialVelocity[i] * normalVector[i];
            }
            
            if (dotProduct > 0) { // Only include positive dot products (etching)
                vectorVelocity = vectorVelocity - potentialVelocity;
            }

            // // Accumulate directional velocities
            // vectorVelocity = vectorVelocity - rateSet.direction * rateSet.directionalVelocity;
        }

        return vectorVelocity;
    }

    // The translation field should be disabled when using a surface model
    // which only depends on an analytic velocity field
    int getTranslationFieldOptions() const override { return 0; }

    // Return direction vector for a specific rateSet
    Vec3D<NumericType> getDirection(const int rateSetId) const override {
        const auto& rateSet = rateSets_[rateSetId];
        return rateSet.direction;
    }

    bool useVisibilities(const int rateSetId) const override { return rateSets_[rateSetId].calculateVisibility; }
    
    bool useVisibilities() const override { return useVisibilities_; }

    int numRates() const override { return rateSets_.size(); }

private:
    bool isMaskMaterial(const int material, const std::vector<int>& maskMaterials) const {
        return std::find(maskMaterials.begin(), maskMaterials.end(), material) != maskMaterials.end();
    }
};

} // namespace impl

/// Directional etching with multiple rate sets and masking materials.
template <typename NumericType, int D>
class DirectionalEtching : public ProcessModel<NumericType, D> {
public:
    struct RateSet {
        Vec3D<NumericType> direction;
        NumericType directionalVelocity;
        NumericType isotropicVelocity;
        std::vector<Material> maskMaterials;
        bool calculateVisibility;

        RateSet(const Vec3D<NumericType>& dir,
                NumericType dirVel,
                NumericType isoVel,
                const std::vector<Material>& masks,
                bool calcVis)
            : direction(dir), directionalVelocity(dirVel),
              isotropicVelocity(isoVel), maskMaterials(masks),
              calculateVisibility(calcVis) {}
    };

    // Constructor accepting single rate set
    DirectionalEtching(const RateSet& rateSet) : DirectionalEtching(std::vector<RateSet>{rateSet}) { }

    // Constructor accepting multiple rate sets
    DirectionalEtching(const std::vector<RateSet>& rateSets) {
        useVisibilities_ = false;
        // Convert RateSet materials from Material enum to int
        for (const auto& rs : rateSets) {
            std::vector<int> maskInts;
            for (const auto& mat : rs.maskMaterials) {
                maskInts.push_back(static_cast<int>(mat));
            }
            impl::RateSet<NumericType, D> internalRateSet(
                rs.direction,
                rs.directionalVelocity,
                rs.isotropicVelocity,
                maskInts,
                rs.calculateVisibility
            );
            useVisibilities_ = useVisibilities_ || rs.calculateVisibility;
            rateSets_.push_back(internalRateSet);
        }
        initialize();
    }

    void disableVisibilityCheck() {
        useVisibilities_ = false;
        initialize();
    }

private:
    void initialize() {
        // Default surface model
        auto surfModel = SmartPointer<SurfaceModel<NumericType>>::New();

        // Velocity field with multiple rate sets
        auto velField =
            SmartPointer<impl::DirectionalEtchVelocityField<NumericType, D>>::New(
                rateSets_, useVisibilities_);

        this->setSurfaceModel(surfModel);
        this->setVelocityField(velField);
        this->setProcessName("DirectionalEtching");
    }

    std::vector<impl::RateSet<NumericType, D>> rateSets_;
    bool useVisibilities_;
};

} // namespace viennaps
