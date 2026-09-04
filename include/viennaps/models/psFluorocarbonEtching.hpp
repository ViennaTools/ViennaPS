#pragma once

#include <cmath>

#include "psFluorocarbonParameters.hpp"
#include "psIonModelUtil.hpp"
#include "psPlasmaEtching.hpp"

#include "../materials/psMaterialMap.hpp"
#include "../process/psProcessModel.hpp"
#include "../psConstants.hpp"
#include "../psUnits.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>

#include <vcLogger.hpp>
#include <vcVectorType.hpp>

namespace viennaps {

using namespace viennacore;

// Parameters from:
// A. LaMagna and G. Garozzo "Factors affecting profile evolution in plasma
// etching of SiO2: Modeling and experimental verification" Journal of the
// Electrochemical Society 150(10) 2003 pp. 1896-1902

namespace impl {

template <typename NumericType, int D>
class FluorocarbonSurfaceModel : public SurfaceModel<NumericType> {
  using SurfaceModel<NumericType>::coverages;
  using SurfaceModel<NumericType>::surfaceData;
  static constexpr double eps = 1e-6;
  const FluorocarbonParameters<NumericType> &p;

  double totalSputterRate_ = 0.;
  double totalIonEnhancedRate_ = 0.;
  double totalChemicalRate_ = 0.;
  double totalPolymerDepositionRate_ = 0.;

public:
  FluorocarbonSurfaceModel(
      const FluorocarbonParameters<NumericType> &parameters)
      : p(parameters) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = PointData<NumericType>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints);
    coverages->insertNextScalarData(cov, "eCoverage");
    coverages->insertNextScalarData(cov, "pCoverage");
    coverages->insertNextScalarData(cov, "peCoverage");
  }

  void initializeSurfaceData(unsigned numGeometryPoints) override {
    if (Logger::hasIntermediate()) {
      if (surfaceData == nullptr) {
        surfaceData = PointData<NumericType>::New();
      } else {
        surfaceData->clear();
      }

      std::vector<NumericType> data(numGeometryPoints, 0.);
      surfaceData->insertNextScalarData(data, "ionEnhancedRate");
      surfaceData->insertNextScalarData(data, "sputterRate");
      surfaceData->insertNextScalarData(data, "chemicalRate");
    }
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<PointData<NumericType>> fluxes,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    const auto numPoints = materialIds.size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    auto ionEnhancedFlux = fluxes->getScalarData("ionEnhancedFlux");
    auto ionSputterFlux = fluxes->getScalarData("ionSputterFlux");
    auto ionpeFlux = fluxes->getScalarData("ionpeFlux");
    auto polyFlux = fluxes->getScalarData("polyFlux");
    fluxes->insertNextScalarData(etchRate, "F_ev");
    auto F_ev_rate = fluxes->getScalarData("F_ev");

    const auto eCoverage = coverages->getScalarData("eCoverage");
    const auto pCoverage = coverages->getScalarData("pCoverage");
    const auto peCoverage = coverages->getScalarData("peCoverage");

    // save the etch rate components for visualization
    std::vector<NumericType> *ieRate = nullptr, *spRate = nullptr,
                             *chRate = nullptr;
    if (Logger::hasIntermediate()) {
      ieRate = surfaceData->getScalarData("ionEnhancedRate");
      spRate = surfaceData->getScalarData("sputterRate");
      chRate = surfaceData->getScalarData("chemicalRate");
      ieRate->resize(numPoints);
      spRate->resize(numPoints);
      chRate->resize(numPoints);
    }

    // The etch rate is calculated in nm/s
    const double unitConversion =
        units::Time::getInstance().convertSecond() /
        units::Length::getInstance().convertNanometer();

    bool etchStop = false;
    const auto polyParams = p.getMaterialParameters(Material::Polymer);
    const auto maskParams = p.getMaterialParameters(Material::Mask);

    double totalSputterRate = 0.;
    double totalIonEnhancedRate = 0.;
    double totalChemicalRate = 0.;

#pragma omp parallel for reduction(|| : etchStop)                              \
    reduction(+ : totalSputterRate, totalIonEnhancedRate, totalChemicalRate)
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] <= p.etchStopDepth || etchStop) {
        etchStop = true;
        continue;
      }

      auto matId = MaterialMap::mapToMaterial(materialIds[i]);
      if (pCoverage->at(i) >= 1.) {
        // Polymer Deposition
        etchRate[i] =
            (1 / polyParams.density) *
            std::max((polyFlux->at(i) * p.polyFlux * polyParams.beta_p -
                      ionpeFlux->at(i) * p.ionFlux * peCoverage->at(i)) *
                         polyParams.A_ie,
                     static_cast<NumericType>(0)) *
            unitConversion;
        assert(etchRate[i] >= 0 && "Negative deposition");

      } else if (matId == Material::Mask) {
        // Mask sputtering
        etchRate[i] = (-1. / maskParams.density) * ionSputterFlux->at(i) *
                      p.ionFlux * maskParams.A_sp * unitConversion;

      } else if (matId == Material::Polymer) {
        // Polymer etching
        etchRate[i] =
            std::min((1. / polyParams.density) *
                         (polyFlux->at(i) * p.polyFlux * polyParams.beta_p -
                          ionpeFlux->at(i) * p.ionFlux * polyParams.A_ie *
                              peCoverage->at(i)),
                     0.) *
            unitConversion;

      } else {
        // Substrate etching
        auto matParams = p.getMaterialParameters(matId);
        NumericType density = matParams.density;

        auto sputterRate = ionSputterFlux->at(i) * p.ionFlux *
                           (1. - eCoverage->at(i)) * matParams.A_sp;
        auto ionEnhancedRate = eCoverage->at(i) * ionEnhancedFlux->at(i) *
                               p.ionFlux * matParams.A_ie;
        auto chemicalRate =
            matParams.K * p.etchantFlux *
            std::exp(-matParams.E_a / (constants::kB * p.temperature)) *
            eCoverage->at(i);

        etchRate[i] = (-1. / density) *
                      (chemicalRate + sputterRate + ionEnhancedRate) *
                      unitConversion;

        if (Logger::hasIntermediate()) {
          chRate->at(i) = chemicalRate;
          spRate->at(i) = sputterRate;
          ieRate->at(i) = ionEnhancedRate;
        }

        totalSputterRate += sputterRate;
        totalIonEnhancedRate += ionEnhancedRate;
        totalChemicalRate += chemicalRate;
      }

      assert(!std::isnan(etchRate[i]) && "etchRate NaN");
    }

    totalSputterRate_ += totalSputterRate;
    totalIonEnhancedRate_ += totalIonEnhancedRate;
    totalChemicalRate_ += totalChemicalRate;

    if (etchStop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      VIENNACORE_LOG_INFO("Etch stop depth reached.");
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(etchRate));
  }

  void updateCoverages(SmartPointer<PointData<NumericType>> fluxes,
                       const std::vector<NumericType> &materialIds) override {

    const auto ionEnhancedFlux = fluxes->getScalarData("ionEnhancedFlux");
    const auto ionpeFlux = fluxes->getScalarData("ionpeFlux");
    const auto polyFlux = fluxes->getScalarData("polyFlux");
    const auto etchantFlux = fluxes->getScalarData("etchantFlux");

    const auto eCoverage = coverages->getScalarData("eCoverage");
    const auto pCoverage = coverages->getScalarData("pCoverage");
    const auto peCoverage = coverages->getScalarData("peCoverage");

    // update coverages based on fluxes
    const auto numPoints = materialIds.size();
    eCoverage->resize(numPoints);
    pCoverage->resize(numPoints);
    peCoverage->resize(numPoints);

    // pe coverage
    const auto polyParams = p.getMaterialParameters(Material::Polymer);
#pragma omp parallel for
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (etchantFlux->at(i) == 0.) {
        peCoverage->at(i) = 0.;
      } else {
        peCoverage->at(i) =
            (etchantFlux->at(i) * p.etchantFlux * polyParams.beta_e) /
            (etchantFlux->at(i) * p.etchantFlux * polyParams.beta_e +
             ionpeFlux->at(i) * p.ionFlux);
      }
      assert(!std::isnan(peCoverage->at(i)) && "peCoverage NaN");
    }

    // polymer coverage
#pragma omp parallel for
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (polyFlux->at(i) < eps) {
        pCoverage->at(i) = 0.;
      } else if (peCoverage->at(i) < eps || ionpeFlux->at(i) < eps) {
        pCoverage->at(i) = 1.;
      } else {
        auto matParams =
            p.getMaterialParameters(MaterialMap::mapToMaterial(materialIds[i]));
        pCoverage->at(i) =
            (polyFlux->at(i) * p.polyFlux * matParams.beta_p) /
            (ionpeFlux->at(i) * p.ionFlux * peCoverage->at(i) + p.delta_p);
      }
      assert(!std::isnan(pCoverage->at(i)) && "pCoverage NaN");
    }

    // etchant coverage
#pragma omp parallel for
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (pCoverage->at(i) < 1.) {
        if (etchantFlux->at(i) == 0.) {
          eCoverage->at(i) = 0;
        } else {
          auto matParams = p.getMaterialParameters(
              MaterialMap::mapToMaterial(materialIds[i]));
          NumericType F_ev =
              matParams.K * p.etchantFlux *
              std::exp(-matParams.E_a / (constants::kB * p.temperature));
          eCoverage->at(i) =
              (etchantFlux->at(i) * p.etchantFlux * matParams.beta_e *
               (1. - pCoverage->at(i))) /
              (p.k_ie * ionEnhancedFlux->at(i) * p.ionFlux + p.k_ev * F_ev +
               etchantFlux->at(i) * p.etchantFlux * matParams.beta_e);
        }
      } else {
        eCoverage->at(i) = 0.;
      }
      assert(!std::isnan(eCoverage->at(i)) && "eCoverage NaN");
    }
  }

  void resetTotalRates() {
    totalSputterRate_ = 0.;
    totalIonEnhancedRate_ = 0.;
    totalChemicalRate_ = 0.;
  }

  void logTotalRates() {
    double totalRate =
        totalSputterRate_ + totalIonEnhancedRate_ + totalChemicalRate_;
    VIENNACORE_LOG_INFO("Substrate etch rate components (normalized):");
    VIENNACORE_LOG_INFO("Chemical: " +
                        std::to_string(totalChemicalRate_ / totalRate));
    VIENNACORE_LOG_INFO("Ion Enhanced: " +
                        std::to_string(totalIonEnhancedRate_ / totalRate));
    VIENNACORE_LOG_INFO("Sputter: " +
                        std::to_string(totalSputterRate_ / totalRate));
  }
};

template <typename NumericType, int D>
class FluorocarbonIon
    : public viennaray::Particle<FluorocarbonIon<NumericType, D>, NumericType> {
  const FluorocarbonParameters<NumericType> &p;
  const NumericType A;
  NumericType minEnergy = std::numeric_limits<NumericType>::max();
  NumericType E = 0.;

public:
  FluorocarbonIon(const FluorocarbonParameters<NumericType> &parameters)
      : p(parameters),
        A(1. / (1. + p.Ions.n_l * (M_PI_2 / p.Ions.inflectAngle - 1.))) {
    for (auto m : p.materials) {
      minEnergy = std::min(minEnergy, m.Eth_ie);
    }
    assert(minEnergy < std::numeric_limits<NumericType>::max());
  }
  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        PointData<NumericType> &localData,
                        const PointData<NumericType> *globalData,
                        RNG &) override final {
    // collect data for this hit
    assert(E >= 0 && "Negative energy ion");

    const auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e-4 && "Error in calculating cos theta");

    auto matParams =
        p.getMaterialParameters(MaterialMap::mapToMaterial(materialId));
    const NumericType B_sp = matParams.B_sp;
    const NumericType Eth_sp = matParams.Eth_sp;
    const NumericType Eth_ie = matParams.Eth_ie;

    const auto sqrtE = std::sqrt(E);

    // sputtering yield Y_s
    localData.addToScalarData(
        0, primID,
        std::max(sqrtE - std::sqrt(Eth_sp), (NumericType)0) *
            (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta);

    // ion enhanced etching yield Y_ie
    localData.addToScalarData(
        1, primID,
        std::max(sqrtE - std::sqrt(Eth_ie), (NumericType)0) * cosTheta);

    // polymer yield Y_p
    if (matParams.id != Material::Polymer)
      matParams = p.getMaterialParameters(Material::Polymer);
    localData.addToScalarData(
        2, primID,
        std::max(sqrtE - std::sqrt(matParams.Eth_ie), (NumericType)0) *
            cosTheta);
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const PointData<NumericType> *globalData,
                    RNG &Rng) override final {

    auto cosTheta = getCosTheta(rayDir, geomNormal);
    NumericType incAngle = std::acos(cosTheta);

    NumericType newEnergy =
        updateEnergy(Rng, E, incAngle, A, p.Ions.inflectAngle, p.Ions.n_l);

    if (newEnergy > minEnergy) {
      E = newEnergy;
      auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, Rng,
          M_PI_2 - std::min(incAngle, p.Ions.minAngle));
      return std::pair<NumericType, Vec3D<NumericType>>{0., direction};
    } else {
      return VIENNARAY_PARTICLE_STOP;
    }
  }
  void initNew(RNG &RNG) override final {
    E = initNormalDistEnergy(RNG, p.Ions.meanEnergy, p.Ions.sigmaEnergy);
  }
  NumericType getSourceDistributionPower() const override final {
    return p.Ions.exponent;
  }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"ionSputterFlux", "ionEnhancedFlux", "ionpeFlux"};
  }
};

template <typename NumericType, int D>
class FluorocarbonNeutral
    : public viennaray::Particle<FluorocarbonNeutral<NumericType, D>,
                                 NumericType> {
  const FluorocarbonParameters<NumericType> &p_;
  const std::string label_;
  const bool isEtchant_ = true; // true for etchant, false for polymer

public:
  FluorocarbonNeutral(const FluorocarbonParameters<NumericType> &parameters,
                      const std::string &label, bool isEtchant = true)
      : p_(parameters), label_(label), isEtchant_(isEtchant) {}
  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int, PointData<NumericType> &localData,
                        const PointData<NumericType> *, RNG &) override final {
    // collect data for this hit
    localData.addToScalarData(0, primID, rayWeight);
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const PointData<NumericType> *globalData,
                    RNG &Rng) override final {
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, Rng);

    const auto &phi_e = globalData->getScalarData(0)->at(primID);
    const auto &phi_p = globalData->getScalarData(1)->at(primID);
    NumericType Seff = std::max(1 - phi_e - phi_p, NumericType(0));

    if (Seff > 0) {
      Seff *=
          isEtchant_
              ? p_.getMaterialParameters(MaterialMap::mapToMaterial(materialId))
                    .beta_e
              : p_.getMaterialParameters(MaterialMap::mapToMaterial(materialId))
                    .beta_p;
    }

    return std::pair<NumericType, Vec3D<NumericType>>{Seff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {label_};
  }
};
} // namespace impl

#ifdef VIENNACORE_COMPILE_GPU
namespace gpu {
template <typename NumericType, int D>
class FluorocarbonEtching : public ProcessModelGPU<NumericType, D> {
  bool initialized = false;

public:
  explicit FluorocarbonEtching(
      const viennaps::FluorocarbonParameters<NumericType> &parameters)
      : params_(parameters) {
    initializeModel();
  }

  ~FluorocarbonEtching() override { this->processData.free(); }

  void setParameters(
      const viennaps::FluorocarbonParameters<NumericType> &parameters) {
    params_ = parameters;
    initializeModel();
  }

  void initialize(SmartPointer<Domain<NumericType, D>> domain,
                  const NumericType processDuration) override {
    if (initialized) {
      return;
    }

    auto surfModel = std::dynamic_pointer_cast<
        impl::FluorocarbonSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());

    if (Logger::hasInfo()) {
      surfModel->resetTotalRates();
    }

    initialized = true;
  }

  void finalize(SmartPointer<Domain<NumericType, D>> domain,
                const NumericType processedDuration) override {
    auto surfModel = std::dynamic_pointer_cast<
        impl::FluorocarbonSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());

    if (Logger::hasInfo()) {
      surfModel->logTotalRates();
    }

    initialized = false;
  }

private:
  viennaps::FluorocarbonParameters<NumericType> params_;
  viennaps::gpu::FluorocarbonParameters gpuParams_;

  void initializeModel() {
    // check if units have been set
    if (units::Length::getInstance().getUnit() == units::Length::UNDEFINED ||
        units::Time::getInstance().getUnit() == units::Time::UNDEFINED) {
      VIENNACORE_LOG_ERROR("Units have not been set.");
    }

    if (params_.materials.empty()) {
      VIENNACORE_LOG_WARNING("No materials have been set in the parameters.");
    }

    gpuParams_ = viennaps::gpu::FluorocarbonParameters(params_);
    this->processData.alloc(sizeof(viennaps::gpu::FluorocarbonParameters));
    this->processData.upload(&gpuParams_, 1);

    // particles
    viennaray::gpu::Particle<NumericType> ion;
    ion.name = "Ion"; // name for shader programs postfix
    ion.dataLabels.push_back("ionSputterFlux");
    ion.dataLabels.push_back("ionEnhancedFlux");
    ion.dataLabels.push_back("ionpeFlux");
    ion.sticking = 0.f;
    ion.cosineExponent = params_.Ions.exponent;

    viennaray::gpu::Particle<NumericType> etchant;
    etchant.name = "Etchant";
    etchant.dataLabels.push_back("etchantFlux");
    etchant.cosineExponent = 1.f;
    for (auto entry : params_.materials) {
      etchant.materialSticking[static_cast<int>(entry.id)] = entry.beta_e;
    }

    viennaray::gpu::Particle<NumericType> poly;
    poly.name = "Polymer";
    poly.dataLabels.push_back("polyFlux");
    poly.cosineExponent = 1.f;
    for (auto entry : params_.materials) {
      poly.materialSticking[static_cast<int>(entry.id)] = entry.beta_p;
    }

    std::unordered_map<std::string, unsigned> pMap = {
        {"Ion", 0}, {"Etchant", 1}, {"Polymer", 2}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__fluorocarbonIonCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__fluorocarbonIonReflection"},
        {0, viennaray::gpu::CallableSlot::INIT,
         "__direct_callable__fluorocarbonIonInit"},
        {1, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__fluorocarbonNeutralCollision"},
        {1, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__fluorocarbonNeutralReflection"},
        {2, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__fluorocarbonNeutralCollision"},
        {2, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__fluorocarbonNeutralReflection"}};
    this->setParticleCallableMap(pMap, cMap);

    // surface model
    auto surfModel =
        SmartPointer<impl::FluorocarbonSurfaceModel<NumericType, D>>::New(
            params_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("FluorocarbonEtching");
    this->getParticleTypes().clear();

    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(poly);

    this->addProcessMetaData(params_);
    this->addUnitsMetaData();

    this->hasGPU = true;
    this->setUseMaterialIds(true);
  }
};

} // namespace gpu
#endif

template <typename NumericType, int D>
class FluorocarbonEtching : public ProcessModelCPU<NumericType, D> {
  bool initialized = false;

public:
  explicit FluorocarbonEtching(
      const FluorocarbonParameters<NumericType> &parameters)
      : params_(parameters) {
    initializeModel();
  }

  void setParameters(const FluorocarbonParameters<NumericType> &parameters) {
    params_ = parameters;
    initializeModel();
  }

  void initialize(SmartPointer<Domain<NumericType, D>> domain,
                  const NumericType processDuration) override {
    if (initialized) {
      return;
    }

    auto surfModel = std::dynamic_pointer_cast<
        impl::FluorocarbonSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());

    if (Logger::hasInfo()) {
      surfModel->resetTotalRates();
    }

    initialized = true;
  }

  void finalize(SmartPointer<Domain<NumericType, D>> domain,
                const NumericType processedDuration) override {
    auto surfModel = std::dynamic_pointer_cast<
        impl::FluorocarbonSurfaceModel<NumericType, D>>(
        this->getSurfaceModel());

    if (Logger::hasInfo()) {
      surfModel->logTotalRates();
    }

    initialized = false;
  }

#ifdef VIENNACORE_COMPILE_GPU
  SmartPointer<ProcessModelBase<NumericType, D>> getGPUModel() override {
    auto model =
        SmartPointer<gpu::FluorocarbonEtching<NumericType, D>>::New(params_);
    model->setProcessName(this->getProcessName().value());
    return model;
  }
#endif

private:
  FluorocarbonParameters<NumericType> params_;

  void initializeModel() {
    // check if units have been set
    if (units::Length::getInstance().getUnit() == units::Length::UNDEFINED ||
        units::Time::getInstance().getUnit() == units::Time::UNDEFINED) {
      VIENNACORE_LOG_ERROR("Units have not been set.");
    }

    if (params_.materials.empty()) {
      VIENNACORE_LOG_WARNING("No materials have been set in the parameters.");
    }

    // particles
    auto ion = std::make_unique<impl::FluorocarbonIon<NumericType, D>>(params_);
    auto etchant = std::make_unique<impl::FluorocarbonNeutral<NumericType, D>>(
        params_, "etchantFlux", true);
    auto poly = std::make_unique<impl::FluorocarbonNeutral<NumericType, D>>(
        params_, "polyFlux", false);

    // surface model
    auto surfModel =
        SmartPointer<impl::FluorocarbonSurfaceModel<NumericType, D>>::New(
            params_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("FluorocarbonEtching");
    this->particles.clear();
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(poly);
    this->addProcessMetaData(params_);
    this->addUnitsMetaData();
    this->hasGPU = true;
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(FluorocarbonEtching)

} // namespace viennaps
