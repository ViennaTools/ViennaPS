#pragma once

#include <cmath>

#include "../process/psProcessModel.hpp"
#include "../psConstants.hpp"
#include "../psMaterials.hpp"
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

template <typename NumericType> struct FluorocarbonParameters {

  struct MaterialParameters {

    Material id = Material::Undefined;

    // density
    NumericType density = 2.2; // 1e22 atoms/cm³

    // sticking
    NumericType beta_p = 0.26;
    NumericType beta_e = 0.9;

    // sputtering coefficients
    NumericType Eth_sp = 18.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
    NumericType A_ie = 0.0361;

    // chemical etching
    NumericType K = 0.002789491704544977;
    NumericType E_a = 0.168; // eV
  };

  std::vector<MaterialParameters> materials;

  // fluxes in (1e15 /cm² /s)
  NumericType ionFlux = 56.;
  NumericType etchantFlux = 500.;
  NumericType polyFlux = 100.;

  NumericType delta_p = 1.;
  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();

  NumericType temperature = 300.; // K
  NumericType k_ie = 2.;
  NumericType k_ev = 2.;

  struct IonType {
    NumericType meanEnergy = 100.; // eV
    NumericType sigmaEnergy = 10.; // eV
    NumericType exponent = 500.;

    NumericType inflectAngle = 1.55334303;
    NumericType n_l = 10.;
    NumericType minAngle = 1.3962634;
  } Ions;

  void addMaterial(const MaterialParameters &material) {
    materials.push_back(material);
  }

  auto
  getMaterialParameters(const Material material) const -> MaterialParameters {
    for (const auto &m : materials) {
      if (m.id == material)
        return m;
    }
    Logger::getInstance()
        .addError("Material " + MaterialMap::getMaterialName(material) +
                  " not found in fluorocarbon parameters.")
        .print();
    return MaterialParameters{};
  }

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<double>> processData;

    processData["ionFlux"] = std::vector<double>{ionFlux};
    processData["etchantFlux"] = std::vector<double>{etchantFlux};
    processData["polymerFlux"] = std::vector<double>{polyFlux};
    processData["delta_p"] = std::vector<double>{delta_p};
    processData["etchStopDepth"] = std::vector<double>{etchStopDepth};
    processData["temperature"] = std::vector<double>{temperature};
    processData["k_ie"] = std::vector<double>{k_ie};
    processData["k_ev"] = std::vector<double>{k_ev};
    processData["Ion MeanEnergy"] = std::vector<double>{Ions.meanEnergy};
    processData["Ion SigmaEnergy"] = std::vector<double>{Ions.sigmaEnergy};
    processData["Ion Exponent"] = std::vector<double>{Ions.exponent};
    processData["Ion InflectAngle"] = std::vector<double>{Ions.inflectAngle};
    processData["Ion n_k"] = std::vector<double>{Ions.n_l};
    processData["Ion MinAngle"] = std::vector<double>{Ions.minAngle};
    for (auto mat : materials) {
      std::string prefix = MaterialMap::getMaterialName(mat.id) + " ";
      processData[prefix + "density"] = std::vector<double>{mat.density};
      processData[prefix + "beta_p"] = std::vector<double>{mat.beta_p};
      processData[prefix + "beta_e"] = std::vector<double>{mat.beta_e};
      processData[prefix + "Eth_sp"] = std::vector<double>{mat.Eth_sp};
      processData[prefix + "Eth_ie"] = std::vector<double>{mat.Eth_ie};
      processData[prefix + "A_sp"] = std::vector<double>{mat.A_sp};
      processData[prefix + "B_sp"] = std::vector<double>{mat.B_sp};
      processData[prefix + "A_ie"] = std::vector<double>{mat.A_ie};
      processData[prefix + "K"] = std::vector<double>{mat.K};
      processData[prefix + "E_a"] = std::vector<double>{mat.E_a};
    }

    return processData;
  }
};

namespace impl {

template <typename NumericType, int D>
class FluorocarbonSurfaceModel : public SurfaceModel<NumericType> {
  using SurfaceModel<NumericType>::coverages;
  using SurfaceModel<NumericType>::surfaceData;
  static constexpr double eps = 1e-6;
  const FluorocarbonParameters<NumericType> &p;

public:
  FluorocarbonSurfaceModel(
      const FluorocarbonParameters<NumericType> &parameters)
      : p(parameters) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = viennals::PointData<NumericType>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints);
    coverages->insertNextScalarData(cov, "eCoverage");
    coverages->insertNextScalarData(cov, "pCoverage");
    coverages->insertNextScalarData(cov, "peCoverage");
  }

  void initializeSurfaceData(unsigned numGeometryPoints) override {
    if (Logger::getLogLevel() >= 3) {
      if (surfaceData == nullptr) {
        surfaceData = viennals::PointData<NumericType>::New();
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
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> fluxes,
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
    if (Logger::getLogLevel() >= 3) {
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

#pragma omp parallel for reduction(|| : etchStop)
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
                      ionpeFlux->at(i) * p.ionFlux * peCoverage->at(i)),
                     (NumericType)0) *
            unitConversion;
        assert(etchRate[i] >= 0 && "Negative deposition");
      } else if (matId == Material::Mask) {
        // Mask sputtering
        etchRate[i] = (-1. / maskParams.density) * ionSputterFlux->at(i) *
                      p.ionFlux * unitConversion;
      } else if (matId == Material::Polymer) {
        auto polyParams = p.getMaterialParameters(Material::Polymer);
        // Etching depo layer
        etchRate[i] =
            std::min((1. / polyParams.density) *
                         (polyFlux->at(i) * p.polyFlux * polyParams.beta_p -
                          ionpeFlux->at(i) * p.ionFlux * peCoverage->at(i)),
                     0.) *
            unitConversion;
      } else {
        auto matParams = p.getMaterialParameters(matId);
        NumericType density = matParams.density;
        NumericType F_ev =
            matParams.K * p.etchantFlux *
            std::exp(-matParams.E_a / (constants::kB * p.temperature));

        etchRate[i] =
            (-1. / density) *
            (F_ev * eCoverage->at(i) +
             ionEnhancedFlux->at(i) * p.ionFlux * eCoverage->at(i) +
             ionSputterFlux->at(i) * p.ionFlux * (1. - eCoverage->at(i))) *
            unitConversion;

        if (Logger::getLogLevel() >= 3) {
          chRate->at(i) = F_ev * eCoverage->at(i);
          spRate->at(i) =
              ionSputterFlux->at(i) * p.ionFlux * (1. - eCoverage->at(i));
          ieRate->at(i) = ionEnhancedFlux->at(i) * p.ionFlux * eCoverage->at(i);
        }
      }

      assert(!std::isnan(etchRate[i]) && "etchRate NaN");
    }

    if (etchStop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      Logger::getInstance().addInfo("Etch stop depth reached.").print();
    }

    return SmartPointer<std::vector<NumericType>>::New(std::move(etchRate));
  }

  void updateCoverages(SmartPointer<viennals::PointData<NumericType>> fluxes,
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
};

template <typename NumericType, int D>
class FluorocarbonIon
    : public viennaray::Particle<FluorocarbonIon<NumericType, D>, NumericType> {
  const FluorocarbonParameters<NumericType> &p;
  const NumericType A;
  NumericType minEnergy = std::numeric_limits<NumericType>::max();
  NumericType E;

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
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *globalData,
                        RNG &) override final {
    // collect data for this hit
    assert(primID < localData.getVectorData(0).size() && "id out of bounds");
    assert(E >= 0 && "Negative energy ion");

    const auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e-4 && "Error in calculating cos theta");

    auto matParams =
        p.getMaterialParameters(MaterialMap::mapToMaterial(materialId));
    const NumericType A_sp = matParams.A_sp;
    const NumericType B_sp = matParams.B_sp;
    const NumericType A_ie = matParams.A_ie;
    const NumericType Eth_sp = matParams.Eth_sp;
    const NumericType Eth_ie = matParams.Eth_ie;

    const auto sqrtE = std::sqrt(E);

    // sputtering yield Y_s
    localData.getVectorData(0)[primID] +=
        A_sp * std::max(sqrtE - std::sqrt(Eth_sp), (NumericType)0) *
        (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;

    // ion enhanced etching yield Y_ie
    localData.getVectorData(1)[primID] +=
        A_ie * std::max(sqrtE - std::sqrt(Eth_ie), (NumericType)0) * cosTheta;

    // polymer yield Y_p
    if (matParams.id != Material::Polymer)
      matParams = p.getMaterialParameters(Material::Polymer);
    localData.getVectorData(2)[primID] +=
        matParams.A_ie *
        std::max(sqrtE - std::sqrt(matParams.Eth_ie), (NumericType)0) *
        cosTheta;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override final {

    // Small incident angles are reflected with the energy fraction centered
    // at
    // 0
    NumericType incAngle = std::acos(-DotProduct(rayDir, geomNormal));
    NumericType Eref_peak;
    if (incAngle >= p.Ions.inflectAngle) {
      Eref_peak =
          1. - (1. - A) * (M_PI_2 - incAngle) / (M_PI_2 - p.Ions.inflectAngle);
    } else {
      Eref_peak = A * std::pow(incAngle / p.Ions.inflectAngle, p.Ions.n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle
    // energy
    NumericType newEnergy;
    std::normal_distribution<NumericType> normalDist(Eref_peak * E, 0.1 * E);
    do {
      newEnergy = normalDist(Rng);
    } while (newEnergy > E || newEnergy < 0.);

    if (newEnergy > minEnergy) {
      E = newEnergy;
      auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, Rng,
          M_PI_2 - std::min(incAngle, p.Ions.minAngle));
      return std::pair<NumericType, Vec3D<NumericType>>{0., direction};
    } else {
      return std::pair<NumericType, Vec3D<NumericType>>{
          1., Vec3D<NumericType>{0., 0., 0.}};
    }
  }
  void initNew(RNG &RNG) override final {
    std::normal_distribution<NumericType> normalDist{p.Ions.meanEnergy,
                                                     p.Ions.sigmaEnergy};
    do {
      E = normalDist(RNG);
    } while (E < minEnergy);
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

public:
  FluorocarbonNeutral(const FluorocarbonParameters<NumericType> &parameters,
                      const std::string &label)
      : p_(parameters), label_(label) {}
  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &,
                        const Vec3D<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override final {
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, Rng);

    const auto &phi_e = globalData->getVectorData(0)[primID];
    const auto &phi_p = globalData->getVectorData(1)[primID];
    NumericType Seff = std::max(1 - phi_e - phi_p, (NumericType)0);

    if (Seff > 0) {
      Seff *= p_.getMaterialParameters(MaterialMap::mapToMaterial(materialId))
                  .beta_e;
    }

    return std::pair<NumericType, Vec3D<NumericType>>{Seff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {label_};
  }
};
} // namespace impl

template <typename NumericType, int D>
class FluorocarbonEtching : public ProcessModelCPU<NumericType, D> {
public:
  explicit FluorocarbonEtching(
      const FluorocarbonParameters<NumericType> &parameters)
      : params_(parameters) {
    initialize();
  }

  void setParameters(const FluorocarbonParameters<NumericType> &parameters) {
    params_ = parameters;
    initialize();
  }

private:
  FluorocarbonParameters<NumericType> params_;

  void initialize() {
    // check if units have been set
    if (units::Length::getInstance().getUnit() == units::Length::UNDEFINED ||
        units::Time::getInstance().getUnit() == units::Time::UNDEFINED) {
      Logger::getInstance().addError("Units have not been set.").print();
    }

    if (params_.materials.empty()) {
      Logger::getInstance()
          .addWarning("No materials have been set in the parameters.")
          .print();
    }

    // particles
    auto ion = std::make_unique<impl::FluorocarbonIon<NumericType, D>>(params_);
    auto etchant = std::make_unique<impl::FluorocarbonNeutral<NumericType, D>>(
        params_, "etchantFlux");
    auto poly = std::make_unique<impl::FluorocarbonNeutral<NumericType, D>>(
        params_, "polyFlux");

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
    this->processMetaData = params_.toProcessMetaData();
    this->processMetaData["Units"] = std::vector<double>{
        static_cast<double>(units::Length::getInstance().getUnit()),
        static_cast<double>(units::Time::getInstance().getUnit())};
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(FluorocarbonEtching)

} // namespace viennaps
