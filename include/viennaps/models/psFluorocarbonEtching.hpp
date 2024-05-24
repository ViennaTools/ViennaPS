#pragma once

#include <cmath>

#include "../psMaterials.hpp"
#include "../psProcessModel.hpp"

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

#include <vcLogger.hpp>
#include <vcVectorUtil.hpp>

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
};

namespace impl {

template <typename NumericType, int D>
class FluorocarbonSurfaceModel : public SurfaceModel<NumericType> {
  using SurfaceModel<NumericType>::coverages;
  static constexpr double eps = 1e-6;
  const FluorocarbonParameters<NumericType> &p;

public:
  FluorocarbonSurfaceModel(
      const FluorocarbonParameters<NumericType> &parameters)
      : p(parameters) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = SmartPointer<lsPointData<NumericType>>::New();
    } else {
      coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints);
    coverages->insertNextScalarData(cov, "eCoverage");
    coverages->insertNextScalarData(cov, "pCoverage");
    coverages->insertNextScalarData(cov, "peCoverage");
  }

  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<lsPointData<NumericType>> rates,
                      const std::vector<Triple<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    updateCoverages(rates, materialIds);
    const auto numPoints = materialIds.size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    auto ionEnhancedRate = rates->getScalarData("ionEnhancedRate");
    auto ionSputteringRate = rates->getScalarData("ionSputteringRate");
    auto ionpeRate = rates->getScalarData("ionpeRate");
    auto polyRate = rates->getScalarData("polyRate");
    rates->insertNextScalarData(etchRate, "F_ev");
    auto F_ev_rate = rates->getScalarData("F_ev");

    const auto eCoverage = coverages->getScalarData("eCoverage");
    const auto pCoverage = coverages->getScalarData("pCoverage");
    const auto peCoverage = coverages->getScalarData("peCoverage");

    bool etchStop = false;

    // calculate etch rates
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] <= p.etchStopDepth) {
        etchStop = true;
        break;
      }

      auto matId = MaterialMap::mapToMaterial(materialIds[i]);
      if (matId == Material::Mask) {
        etchRate[i] = (-1. / p.Mask.rho) * ionSputteringRate->at(i) * p.ionFlux;
      } else if (pCoverage->at(i) >= 1.) {
        // Deposition
        etchRate[i] =
            (1 / p.Polymer.rho) *
            std::max((polyRate->at(i) * p.polyFlux * p.beta_p -
                      ionpeRate->at(i) * p.ionFlux * peCoverage->at(i)),
                     (NumericType)0);
        assert(etchRate[i] >= 0 && "Negative deposition");
      } else if (matId == Material::Polymer) {
        // Etching depo layer
        etchRate[i] =
            std::min((1 / p.Polymer.rho) *
                         (polyRate->at(i) * p.polyFlux * p.beta_p -
                          ionpeRate->at(i) * p.ionFlux * peCoverage->at(i)),
                     (NumericType)0);
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
             ionEnhancedRate->at(i) * p.ionFlux * eCoverage->at(i) +
             ionSputteringRate->at(i) * p.ionFlux * (1. - eCoverage->at(i)));

        F_ev_rate->at(i) = F_ev * eCoverage->at(i);
        ionSputteringRate->at(i) =
            ionSputteringRate->at(i) * p.ionFlux * (1. - eCoverage->at(i));
        ionEnhancedRate->at(i) =
            ionEnhancedRate->at(i) * p.ionFlux * eCoverage->at(i);
      }

      // etch rate is in nm / s

      assert(!std::isnan(etchRate[i]) && "etchRate NaN");
    }

    if (etchStop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      Logger::getInstance().addInfo("Etch stop depth reached.").print();
    }

    return SmartPointer<std::vector<NumericType>>::New(etchRate);
  }

  void updateCoverages(SmartPointer<lsPointData<NumericType>> rates,
                       const std::vector<NumericType> &materialIds) override {

    const auto ionEnhancedRate = rates->getScalarData("ionEnhancedRate");
    const auto ionpeRate = rates->getScalarData("ionpeRate");
    const auto polyRate = rates->getScalarData("polyRate");
    const auto etchantRate = rates->getScalarData("etchantRate");

    const auto eCoverage = coverages->getScalarData("eCoverage");
    const auto pCoverage = coverages->getScalarData("pCoverage");
    const auto peCoverage = coverages->getScalarData("peCoverage");

    // update coverages based on fluxes
    const auto numPoints = ionEnhancedRate->size();
    eCoverage->resize(numPoints);
    pCoverage->resize(numPoints);
    peCoverage->resize(numPoints);

    // pe coverage
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (etchantRate->at(i) == 0.) {
        peCoverage->at(i) = 0.;
      } else {
        peCoverage->at(i) = (etchantRate->at(i) * p.etchantFlux * p.beta_pe) /
                            (etchantRate->at(i) * p.etchantFlux * p.beta_pe +
                             ionpeRate->at(i) * p.ionFlux);
      }
      assert(!std::isnan(peCoverage->at(i)) && "peCoverage NaN");
    }

    // polymer coverage
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (polyRate->at(i) < eps) {
        pCoverage->at(i) = 0.;
      } else if (peCoverage->at(i) < eps || ionpeRate->at(i) < eps) {
        pCoverage->at(i) = 1.;
      } else {
        pCoverage->at(i) =
            (polyRate->at(i) * p.polyFlux * p.beta_p) /
            (ionpeRate->at(i) * p.ionFlux * peCoverage->at(i) + p.delta_p);
      }
      assert(!std::isnan(pCoverage->at(i)) && "pCoverage NaN");
    }

    // etchant coverage
    for (std::size_t i = 0; i < numPoints; ++i) {
      if (pCoverage->at(i) < 1.) {
        if (etchantRate->at(i) == 0.) {
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
              (etchantRate->at(i) * p.etchantFlux * p.beta_e *
               (1. - pCoverage->at(i))) /
              (p.k_ie * ionEnhancedRate->at(i) * p.ionFlux + p.k_ev * F_ev +
               etchantRate->at(i) * p.etchantFlux * p.beta_e);
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
  void surfaceCollision(NumericType rayWeight,
                        const Triple<NumericType> &rayDir,
                        const Triple<NumericType> &geomNormal,
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
  std::pair<NumericType, Triple<NumericType>>
  surfaceReflection(NumericType rayWeight, const Triple<NumericType> &rayDir,
                    const Triple<NumericType> &geomNormal,
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
      return std::pair<NumericType, Triple<NumericType>>{0., direction};
    } else {
      return std::pair<NumericType, Triple<NumericType>>{
          1., Triple<NumericType>{0., 0., 0.}};
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
    return {"ionSputteringRate", "ionEnhancedRate", "ionpeRate"};
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
  void surfaceCollision(NumericType rayWeight, const Triple<NumericType> &,
                        const Triple<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Triple<NumericType>>
  surfaceReflection(NumericType, const Triple<NumericType> &,
                    const Triple<NumericType> &geomNormal,
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
    return std::pair<NumericType, Triple<NumericType>>{stick, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"polyRate"};
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
  void surfaceCollision(NumericType rayWeight, const Triple<NumericType> &,
                        const Triple<NumericType> &, const unsigned int primID,
                        const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<NumericType, Triple<NumericType>>
  surfaceReflection(NumericType rayWeight, const Triple<NumericType> &rayDir,
                    const Triple<NumericType> &geomNormal,
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

    return std::pair<NumericType, Triple<NumericType>>{Seff, direction};
  }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"etchantRate"};
  }
};
} // namespace impl

template <typename NumericType, int D>
class FluorocarbonEtching : public ProcessModel<NumericType, D> {
public:
  FluorocarbonEtching() { initialize(); }
  FluorocarbonEtching(const double ionFlux, const double etchantFlux,
                      const double polyFlux, const NumericType meanEnergy,
                      const NumericType sigmaEnergy,
                      const NumericType exponent = 100.,
                      const NumericType deltaP = 0.,
                      const NumericType etchStopDepth =
                          std::numeric_limits<NumericType>::lowest()) {
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
  }

private:
  FluorocarbonParameters<NumericType> params_;

  void initialize() {
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
    auto velField = SmartPointer<DefaultVelocityField<NumericType>>::New(2);

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("FluorocarbonEtching");
    this->insertNextParticleType(ion);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(poly);
  }
};

} // namespace viennaps
