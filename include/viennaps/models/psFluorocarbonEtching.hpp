#pragma once

#include <cmath>

#include "../process/psProcessModel.hpp"
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
  // fluxes in (1e15 /cm² /s)
  NumericType ionFlux = 56.;
  NumericType etchantFlux = 500.;
  NumericType polyFlux = 100.;

  NumericType delta_p = 1.;
  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();

  NumericType temperature = 300.; // K
  NumericType k_ie = 2.;
  NumericType k_ev = 2.;

  NumericType beta_pe = 0.6;
  NumericType beta_p = 0.26;
  NumericType beta_e = 0.9;

  // Mask
  struct MaskType {
    NumericType rho = 500.; // 1e22 atoms/cm³
    NumericType beta_p = 0.01;
    NumericType beta_e = 0.1;

    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
    NumericType Eth_sp = 20.; // eV
  } Mask;

  // SiO2
  struct SiO2Type {
    // density
    NumericType rho = 2.2; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 18.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
    NumericType A_ie = 0.0361;

    // chemical etching
    NumericType K = 0.002789491704544977;
    NumericType E_a = 0.168; // eV
  } SiO2;

  // Polymer
  struct PolymerType {
    NumericType rho = 2.; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_ie = 4.; // eV
    NumericType A_ie = 0.0361 * 4;
  } Polymer;

  // Si3N4
  struct Si3N4Type {
    // density
    NumericType rho = 2.3; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 18.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp = 0.0139;
    NumericType B_sp = 9.3;
    NumericType A_ie = 0.0361;

    // chemical etching
    NumericType K = 0.002789491704544977;
    NumericType E_a = 0.168; // eV
  } Si3N4;

  // Si
  struct SiType {
    // density
    NumericType rho = 5.02; // 1e22 atoms/cm³

    // sputtering coefficients
    NumericType Eth_sp = 20.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp = 0.0337;
    NumericType B_sp = 9.3;
    NumericType A_ie = 0.0361;

    // chemical etching
    NumericType K = 0.029997010728956663;
    NumericType E_a = 0.108; // eV
  } Si;

  struct IonType {
    NumericType meanEnergy = 100.; // eV
    NumericType sigmaEnergy = 10.; // eV
    NumericType exponent = 500.;

    NumericType inflectAngle = 1.55334303;
    NumericType n_l = 10.;
    NumericType minAngle = 1.3962634;
  } Ions;

  // fixed
  static constexpr double kB = 8.617333262 * 1e-5; // eV / K

  auto toProcessMetaData() const {
    std::unordered_map<std::string, std::vector<NumericType>> processData;

    processData["ionFlux"] = std::vector<NumericType>{ionFlux};
    processData["etchantFlux"] = std::vector<NumericType>{etchantFlux};
    processData["polymerFlux"] = std::vector<NumericType>{polyFlux};
    processData["delta_p"] = std::vector<NumericType>{delta_p};
    processData["etchStopDepth"] = std::vector<NumericType>{etchStopDepth};
    processData["temperature"] = std::vector<NumericType>{temperature};
    processData["k_ie"] = std::vector<NumericType>{k_ie};
    processData["k_ev"] = std::vector<NumericType>{k_ev};
    processData["beta_pe"] = std::vector<NumericType>{beta_pe};
    processData["beta_p"] = std::vector<NumericType>{beta_p};
    processData["beta_e"] = std::vector<NumericType>{beta_e};
    processData["Mask Rho"] = std::vector<NumericType>{Mask.rho};
    processData["Mask Beta_P"] = std::vector<NumericType>{Mask.beta_p};
    processData["Mask Beta_E"] = std::vector<NumericType>{Mask.beta_e};
    processData["Mask A_sp"] = std::vector<NumericType>{Mask.A_sp};
    processData["Mask B_sp"] = std::vector<NumericType>{Mask.B_sp};
    processData["Mask E_th_sp"] = std::vector<NumericType>{Mask.Eth_sp};
    processData["SiO2 Rho"] = std::vector<NumericType>{SiO2.rho};
    processData["SiO2 E_th_sp"] = std::vector<NumericType>{SiO2.Eth_sp};
    processData["SiO2 E_th_ie"] = std::vector<NumericType>{SiO2.Eth_ie};
    processData["SiO2 A_Sp"] = std::vector<NumericType>{SiO2.A_sp};
    processData["SiO2 B_sp"] = std::vector<NumericType>{SiO2.B_sp};
    processData["SiO2 K"] = std::vector<NumericType>{SiO2.K};
    processData["SiO2 E_a"] = std::vector<NumericType>{SiO2.E_a};
    processData["Polymer Rho"] = std::vector<NumericType>{Polymer.rho};
    processData["Polymer E_th_ie"] = std::vector<NumericType>{Polymer.Eth_ie};
    processData["Polymer A_ie"] = std::vector<NumericType>{Polymer.A_ie};
    processData["Si3N4 Rho"] = std::vector<NumericType>{Si3N4.rho};
    processData["Si3N4 E_th_sp"] = std::vector<NumericType>{Si3N4.Eth_sp};
    processData["Si3N4 E_th_ie"] = std::vector<NumericType>{Si3N4.Eth_ie};
    processData["Si3N4 A_Sp"] = std::vector<NumericType>{Si3N4.A_sp};
    processData["Si3N4 B_sp"] = std::vector<NumericType>{Si3N4.B_sp};
    processData["Si3N4 K"] = std::vector<NumericType>{Si3N4.K};
    processData["Si3N4 EA"] = std::vector<NumericType>{Si3N4.E_a};
    processData["Si Rho"] = std::vector<NumericType>{Si.rho};
    processData["Si E_th_sp"] = std::vector<NumericType>{Si.Eth_sp};
    processData["Si E_th_ie"] = std::vector<NumericType>{Si.Eth_ie};
    processData["Si A_Sp"] = std::vector<NumericType>{Si.A_sp};
    processData["Si B_sp"] = std::vector<NumericType>{Si.B_sp};
    processData["Si K"] = std::vector<NumericType>{Si.K};
    processData["Si E_a"] = std::vector<NumericType>{Si.E_a};
    processData["Ion MeanEnergy"] = std::vector<NumericType>{Ions.meanEnergy};
    processData["Ion SigmaEnergy"] = std::vector<NumericType>{Ions.sigmaEnergy};
    processData["Ion Exponent"] = std::vector<NumericType>{Ions.exponent};
    processData["Ion InflectAngle"] =
        std::vector<NumericType>{Ions.inflectAngle};
    processData["Ion n_k"] = std::vector<NumericType>{Ions.n_l};
    processData["Ion MinAngle"] = std::vector<NumericType>{Ions.minAngle};

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
    if (Logger::getLogLevel() > 3) {
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
    if (Logger::getLogLevel() > 3) {
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

#pragma omp parallel for reduction(|| : etchStop)
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] <= p.etchStopDepth || etchStop) {
        etchStop = true;
        continue;
      }

      auto matId = MaterialMap::mapToMaterial(materialIds[i]);
      if (matId == Material::Mask) {
        etchRate[i] = (-1. / p.Mask.rho) * ionSputterFlux->at(i) * p.ionFlux *
                      unitConversion;
      } else if (pCoverage->at(i) >= 1.) {
        // Deposition
        etchRate[i] =
            (1 / p.Polymer.rho) *
            std::max((polyFlux->at(i) * p.polyFlux * p.beta_p -
                      ionpeFlux->at(i) * p.ionFlux * peCoverage->at(i)),
                     (NumericType)0) *
            unitConversion;
        assert(etchRate[i] >= 0 && "Negative deposition");
      } else if (matId == Material::Polymer) {
        // Etching depo layer
        etchRate[i] =
            std::min((1 / p.Polymer.rho) *
                         (polyFlux->at(i) * p.polyFlux * p.beta_p -
                          ionpeFlux->at(i) * p.ionFlux * peCoverage->at(i)),
                     (NumericType)0) *
            unitConversion;
      } else {
        NumericType density = 1.;
        NumericType F_ev = 0.;
        switch (matId) {
        case Material::Si: {

          density = -p.Si.rho;
          F_ev = p.Si.K * p.etchantFlux *
                 std::exp(-p.Si.E_a / (FluorocarbonParameters<NumericType>::kB *
                                       p.temperature));
          break;
        }
        case Material::SiO2: {

          F_ev =
              p.SiO2.K * p.etchantFlux *
              std::exp(-p.SiO2.E_a / (FluorocarbonParameters<NumericType>::kB *
                                      p.temperature));
          density = -p.SiO2.rho;
          break;
        }
        case Material::Si3N4: {

          F_ev =
              p.Si3N4.K * p.etchantFlux *
              std::exp(-p.Si3N4.E_a / (FluorocarbonParameters<NumericType>::kB *
                                       p.temperature));
          density = -p.Si3N4.rho;
          break;
        }
        default:
          break;
        }

        etchRate[i] =
            (1 / density) *
            (F_ev * eCoverage->at(i) +
             ionEnhancedFlux->at(i) * p.ionFlux * eCoverage->at(i) +
             ionSputterFlux->at(i) * p.ionFlux * (1. - eCoverage->at(i))) *
            unitConversion;

        if (Logger::getLogLevel() > 3) {
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
#pragma omp parallel for
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (etchantFlux->at(i) == 0.) {
        peCoverage->at(i) = 0.;
      } else {
        peCoverage->at(i) = (etchantFlux->at(i) * p.etchantFlux * p.beta_pe) /
                            (etchantFlux->at(i) * p.etchantFlux * p.beta_pe +
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
        pCoverage->at(i) =
            (polyFlux->at(i) * p.polyFlux * p.beta_p) /
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
          NumericType F_ev = 0.;
          switch (MaterialMap::mapToMaterial(materialIds[i])) {
          case Material::Si:
            F_ev =
                p.Si.K * p.etchantFlux *
                std::exp(-p.Si.E_a / (FluorocarbonParameters<NumericType>::kB *
                                      p.temperature));
            break;
          case Material::SiO2:
            F_ev = p.SiO2.K * p.etchantFlux *
                   std::exp(-p.SiO2.E_a /
                            (FluorocarbonParameters<NumericType>::kB *
                             p.temperature));
            break;
          case Material::Si3N4:
            F_ev = p.Si3N4.K * p.etchantFlux *
                   std::exp(-p.Si3N4.E_a /
                            (FluorocarbonParameters<NumericType>::kB *
                             p.temperature));
            break;
          default:
            F_ev = 0.;
          }
          eCoverage->at(i) =
              (etchantFlux->at(i) * p.etchantFlux * p.beta_e *
               (1. - pCoverage->at(i))) /
              (p.k_ie * ionEnhancedFlux->at(i) * p.ionFlux + p.k_ev * F_ev +
               etchantFlux->at(i) * p.etchantFlux * p.beta_e);
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
  const NumericType minEnergy;
  NumericType E;

public:
  FluorocarbonIon(const FluorocarbonParameters<NumericType> &parameters)
      : p(parameters),
        A(1. / (1. + p.Ions.n_l * (M_PI_2 / p.Ions.inflectAngle - 1.))),
        minEnergy(std::min({p.Si.Eth_ie, p.SiO2.Eth_ie, p.Si3N4.Eth_ie})) {}
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
    assert(cosTheta <= 1 + 4 && "Error in calculating cos theta");

    NumericType A_sp = 0.;
    NumericType B_sp = 1.;
    NumericType A_ie = 0.;
    NumericType Eth_sp = 1.;
    NumericType Eth_ie = 1.;
    switch (MaterialMap::mapToMaterial(materialId)) {
    case Material::Si:
      A_sp = p.Si.A_sp;
      B_sp = p.Si.B_sp;
      A_ie = p.Si.A_ie;
      Eth_sp = p.Si.Eth_sp;
      Eth_ie = p.Si.Eth_ie;
      break;
    case Material::SiO2:
      A_sp = p.SiO2.A_sp;
      B_sp = p.SiO2.B_sp;
      A_ie = p.SiO2.A_ie;
      Eth_sp = p.SiO2.Eth_sp;
      Eth_ie = p.SiO2.Eth_ie;
      break;
    case Material::Si3N4:
      A_sp = p.Si3N4.A_sp;
      B_sp = p.Si3N4.B_sp;
      A_ie = p.Si3N4.A_ie;
      Eth_sp = p.Si3N4.Eth_sp;
      Eth_ie = p.Si3N4.Eth_ie;
      break;
    case Material::Polymer:
      A_sp = p.Polymer.A_ie;
      B_sp = 1.;
      A_ie = p.Polymer.A_ie;
      Eth_sp = p.Polymer.Eth_ie;
      Eth_ie = p.Polymer.Eth_ie;
      break;
    default:
      A_sp = 0.;
      B_sp = 0.;
      A_ie = 0.;
      Eth_sp = 0.;
      Eth_ie = 0.;
    }

    const auto sqrtE = std::sqrt(E);

    // sputtering yield Y_s
    localData.getVectorData(0)[primID] +=
        A_sp * std::max(sqrtE - std::sqrt(Eth_sp), (NumericType)0) *
        (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;

    // ion enhanced etching yield Y_ie
    localData.getVectorData(1)[primID] +=
        A_ie * std::max(sqrtE - std::sqrt(Eth_ie), (NumericType)0) * cosTheta;

    // polymer yield Y_p
    localData.getVectorData(2)[primID] +=
        p.Polymer.A_ie *
        std::max(sqrtE - std::sqrt(p.Polymer.Eth_ie), (NumericType)0) *
        cosTheta;
  }
  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override final {

    // Small incident angles are reflected with the energy fraction centered at
    // 0
    NumericType incAngle = std::acos(-DotProduct(rayDir, geomNormal));
    NumericType Eref_peak;
    if (incAngle >= p.Ions.inflectAngle) {
      Eref_peak =
          1. - (1. - A) * (M_PI_2 - incAngle) / (M_PI_2 - p.Ions.inflectAngle);
    } else {
      Eref_peak = A * std::pow(incAngle / p.Ions.inflectAngle, p.Ions.n_l);
    }
    // Gaussian distribution around the Eref_peak scaled by the particle energy
    NumericType newEnergy;
    std::normal_distribution<NumericType> normalDist(Eref_peak * E, 0.1 * E);
    do {
      newEnergy = normalDist(Rng);
    } while (newEnergy > E || newEnergy < 0.);

    if (newEnergy > minEnergy) {
      E = newEnergy;
      auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, Rng, std::max(incAngle, p.Ions.minAngle));
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
class FluorocarbonPolymer
    : public viennaray::Particle<FluorocarbonPolymer<NumericType, D>,
                                 NumericType> {
  const FluorocarbonParameters<NumericType> &p;

public:
  FluorocarbonPolymer(const FluorocarbonParameters<NumericType> &parameters)
      : p(parameters) {}
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
  surfaceReflection(NumericType, const Vec3D<NumericType> &,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &Rng) override final {
    auto direction =
        viennaray::ReflectionDiffuse<NumericType, D>(geomNormal, Rng);

    const auto &phi_e = globalData->getVectorData(0)[primID];
    const auto &phi_p = globalData->getVectorData(1)[primID];
    const auto &phi_pe = globalData->getVectorData(2)[primID];

    NumericType stick = 1.;
    if (MaterialMap::isMaterial(materialId, Material::Mask))
      stick = p.Mask.beta_p;
    else
      stick = p.beta_p;
    stick *= std::max(1 - phi_e - phi_p, (NumericType)0);
    return std::pair<NumericType, Vec3D<NumericType>>{stick, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"polyFlux"};
  }
};

template <typename NumericType, int D>
class FluorocarbonEtchant
    : public viennaray::Particle<FluorocarbonEtchant<NumericType, D>,
                                 NumericType> {
  const FluorocarbonParameters<NumericType> &p;

public:
  FluorocarbonEtchant(const FluorocarbonParameters<NumericType> &parameters)
      : p(parameters) {}
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
    const auto &phi_pe = globalData->getVectorData(2)[primID];

    NumericType Seff;
    if (MaterialMap::isMaterial(materialId, Material::Mask)) {
      Seff = p.Mask.beta_p * std::max(1 - phi_e - phi_p, (NumericType)0);
    } else if (MaterialMap::isMaterial(materialId, Material::Polymer)) {
      Seff = p.beta_pe * std::max(1 - phi_pe, (NumericType)0);
    } else {
      Seff = p.beta_e * std::max(1 - phi_e - phi_p, (NumericType)0);
    }

    return std::pair<NumericType, Vec3D<NumericType>>{Seff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"etchantFlux"};
  }
};
} // namespace impl

template <typename NumericType, int D>
class FluorocarbonEtching : public ProcessModel<NumericType, D> {
public:
  FluorocarbonEtching() { initialize(); }
  FluorocarbonEtching(
      double ionFlux, double etchantFlux, double polyFlux,
      NumericType meanEnergy, NumericType sigmaEnergy,
      NumericType exponent = 100., NumericType deltaP = 0.,
      NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest()) {
    params_.ionFlux = ionFlux;
    params_.etchantFlux = etchantFlux;
    params_.polyFlux = polyFlux;
    params_.Ions.meanEnergy = meanEnergy;
    params_.Ions.sigmaEnergy = sigmaEnergy;
    params_.Ions.exponent = exponent;
    params_.delta_p = deltaP;
    params_.etchStopDepth = etchStopDepth;
    initialize();
  }
  FluorocarbonEtching(const FluorocarbonParameters<NumericType> &parameters)
      : params_(parameters) {
    initialize();
  }

  FluorocarbonParameters<NumericType> &getParameters() { return params_; }
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

    // particles
    auto ion = std::make_unique<impl::FluorocarbonIon<NumericType, D>>(params_);
    auto etchant =
        std::make_unique<impl::FluorocarbonEtchant<NumericType, D>>(params_);
    auto poly =
        std::make_unique<impl::FluorocarbonPolymer<NumericType, D>>(params_);

    // surface model
    auto surfModel =
        SmartPointer<impl::FluorocarbonSurfaceModel<NumericType, D>>::New(
            params_);

    // velocity field
    auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("FluorocarbonEtching");
    this->particles.clear();
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(poly);
    this->processMetaData = params_.toProcessMetaData();
    this->processMetaData["Units"] = std::vector<NumericType>{
        static_cast<NumericType>(units::Length::getInstance().getUnit()),
        static_cast<NumericType>(units::Time::getInstance().getUnit())};
  }
};

PS_PRECOMPILE_PRECISION_DIMENSION(FluorocarbonEtching)

} // namespace viennaps
