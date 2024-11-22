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

    RateSet(const Vec3D<NumericType>& dir,
            NumericType dirVel,
            NumericType isoVel,
            const std::vector<int>& masks)
        : direction(dir), directionalVelocity(dirVel),
          isotropicVelocity(isoVel), maskMaterials(masks) {}
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
            scalarVelocity -= rateSet.isotropicVelocity;
        }

        return scalarVelocity;
    }

    Vec3D<NumericType> getVectorVelocity(const Vec3D<NumericType>& coordinate,
                                         int material,
                                         const Vec3D<NumericType>& normalVector,
                                         unsigned long pointId) override {
        Vec3D<NumericType> vectorVelocity = {0.0, 0.0, 0.0};

        for (const auto& rateSet : rateSets_) {
            if (isMaskMaterial(material, rateSet.maskMaterials)) {
                continue; // Skip this rate set if material is masked
            }
            if (useVisibilities_ && this->visibilities_->at(pointId) == 0.) {
                continue; // Skip if visibility check fails
            }
            // Accumulate directional velocities
            vectorVelocity = vectorVelocity + rateSet.direction * rateSet.directionalVelocity;
        }

        return vectorVelocity;
    }

    // The translation field should be disabled when using a surface model
    // which only depends on an analytic velocity field
    int getTranslationFieldOptions() const override { return 0; }

    bool useVisibilities() const override { return useVisibilities_; }

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

        RateSet(const Vec3D<NumericType>& dir,
                NumericType dirVel,
                NumericType isoVel,
                const std::vector<Material>& masks)
            : direction(dir), directionalVelocity(dirVel),
              isotropicVelocity(isoVel), maskMaterials(masks) {}
    };

    /// Constructor accepting multiple rate sets
    DirectionalEtching(const std::vector<RateSet>& rateSets,
                      const bool useVisibilities = true)
        : useVisibilities_(useVisibilities) {
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
                maskInts
            );
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
