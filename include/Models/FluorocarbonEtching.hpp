#pragma once

#include <cmath>

#include <psLogger.hpp>
#include <psMaterials.hpp>
#include <psProcessModel.hpp>

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

template <typename NumericType, int D>
class FluorocarbonSurfaceModel : public psSurfaceModel<NumericType> {
  using psSurfaceModel<NumericType>::Coverages;
  static constexpr double eps = 1e-6;

public:
  FluorocarbonSurfaceModel(const NumericType ionFlux,
                           const NumericType etchantFlux,
                           const NumericType polyFlux,
                           const NumericType passedEtchStopDepth)
      : totalIonFlux(ionFlux), totalEtchantFlux(etchantFlux),
        totalPolyFlux(polyFlux),
        F_ev(2.7 * etchantFlux * std::exp(-0.168 / (kB * temperature))),
        etchStopDepth(passedEtchStopDepth) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (Coverages == nullptr) {
      Coverages = psSmartPointer<psPointData<NumericType>>::New();
    } else {
      Coverages->clear();
    }
    std::vector<NumericType> cov(numGeometryPoints);
    Coverages->insertNextScalarData(cov, "eCoverage");
    Coverages->insertNextScalarData(cov, "pCoverage");
    Coverages->insertNextScalarData(cov, "peCoverage");
  }

  psSmartPointer<std::vector<NumericType>> calculateVelocities(
      psSmartPointer<psPointData<NumericType>> Rates,
      const std::vector<std::array<NumericType, 3>> &coordinates,
      const std::vector<NumericType> &materialIds) override {
    updateCoverages(Rates);
    const auto numPoints = materialIds.size();
    std::vector<NumericType> etchRate(numPoints, 0.);

    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto ionSputteringRate = Rates->getScalarData("ionSputteringRate");
    const auto ionpeRate = Rates->getScalarData("ionpeRate");
    const auto polyRate = Rates->getScalarData("polyRate");

    const auto eCoverage = Coverages->getScalarData("eCoverage");
    const auto pCoverage = Coverages->getScalarData("pCoverage");
    const auto peCoverage = Coverages->getScalarData("peCoverage");

    bool etchStop = false;

    // calculate etch rates
    for (size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] <= etchStopDepth) {
        etchStop = true;
        break;
      }

      auto matId =
          psMaterialMap::mapToMaterial(static_cast<int>(materialIds[i]));
      assert(matId == psMaterial::Mask || matId == psMaterial::Polymer ||
             matId == psMaterial::Si || matId == psMaterial::SiO2 ||
             matId == psMaterial::Si3N4 && "Unexptected material");
      if (matId == psMaterial::Mask)
        continue;
      if (pCoverage->at(i) >= 1.) {
        assert(pCoverage->at(i) == 1. && "Correctness assumption");
        // Deposition
        etchRate[i] =
            (1 / rho_p) * (polyRate->at(i) * totalPolyFlux -
                           ionpeRate->at(i) * totalIonFlux * peCoverage->at(i));
        assert(etchRate[i] >= 0 && "Negative deposition");
      } else if (matId == psMaterial::Polymer) {
        // Etching depo layer
        etchRate[i] = std::min(
            (1 / rho_p) * (polyRate->at(i) * totalPolyFlux -
                           ionpeRate->at(i) * totalIonFlux * peCoverage->at(i)),
            0.);
      } else {
        NumericType mat_density = 0;
        if (matId == psMaterial::Si) // crystalline Si at the bottom
        {
          mat_density = -rho_Si;
        } else if (matId == psMaterial::SiO2) // Etching SiO2
        {
          mat_density = -rho_SiO2;
        } else if (matId == psMaterial::Si3N4) // Etching SiNx
        {
          mat_density = -rho_SiNx;
        }
        etchRate[i] =
            (1 / mat_density) *
            (F_ev * eCoverage->at(i) +
             ionEnhancedRate->at(i) * totalIonFlux * eCoverage->at(i) +
             ionSputteringRate->at(i) * totalIonFlux * (1 - eCoverage->at(i)));
      }

      if (std::isnan(etchRate[i])) {
        std::cout << "Error in calculating etch rate at point x = "
                  << coordinates[i][0] << ", y = " << coordinates[i][1]
                  << ", z = " << coordinates[i][2] << std::endl;
        std::cout << "Material: " << static_cast<int>(matId) << std::endl;
        std::cout << "Rates and coverages at this point:\neCoverage: "
                  << eCoverage->at(i) << "\npCoverage: " << pCoverage->at(i)
                  << "\npeCoverage: " << peCoverage->at(i)
                  << "\nionEnhancedRate: " << ionEnhancedRate->at(i)
                  << "\nionSputteringRate: " << ionSputteringRate->at(i)
                  << "\nionpeRate: " << ionpeRate->at(i)
                  << "\npolyRate: " << polyRate->at(i) << std::endl;
      }

      // etch rate is in cm / s
      etchRate[i] *= 1e7; // to convert to nm / s

      assert(!std::isnan(etchRate[i]) && "etchRate NaN");
    }

    if (etchStop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      psLogger::getInstance().addInfo("Etch stop depth reached.").print();
    }

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }

  void
  updateCoverages(psSmartPointer<psPointData<NumericType>> Rates) override {

    const auto ionEnhancedRate = Rates->getScalarData("ionEnhancedRate");
    const auto ionpeRate = Rates->getScalarData("ionpeRate");
    const auto polyRate = Rates->getScalarData("polyRate");
    const auto etchantRate = Rates->getScalarData("etchantRate");
    const auto etchantOnPolyRate = Rates->getScalarData("etchantOnPolyRate");

    const auto eCoverage = Coverages->getScalarData("eCoverage");
    const auto pCoverage = Coverages->getScalarData("pCoverage");
    const auto peCoverage = Coverages->getScalarData("peCoverage");

    // update coverages based on fluxes
    const auto numPoints = ionEnhancedRate->size();
    eCoverage->resize(numPoints);
    pCoverage->resize(numPoints);
    peCoverage->resize(numPoints);

    // pe coverage
    for (size_t i = 0; i < numPoints; ++i) {
      if (etchantOnPolyRate->at(i) == 0.) {
        peCoverage->at(i) = 0.;
      } else {
        peCoverage->at(i) = (etchantOnPolyRate->at(i) * totalEtchantFlux) /
                            (etchantOnPolyRate->at(i) * totalEtchantFlux +
                             ionpeRate->at(i) * totalIonFlux);
      }
      assert(!std::isnan(peCoverage->at(i)) && "peCoverage NaN");
    }

    // polymer coverage
    for (size_t i = 0; i < numPoints; ++i) {
      if (polyRate->at(i) == 0.) {
        pCoverage->at(i) = 0.;
      } else if (peCoverage->at(i) < eps || ionpeRate->at(i) < eps) {
        pCoverage->at(i) = 1.;
      } else {
        pCoverage->at(i) =
            std::min((polyRate->at(i) * totalPolyFlux - delta_p) /
                         (ionpeRate->at(i) * totalIonFlux * peCoverage->at(i)),
                     1.);
      }
      assert(!std::isnan(pCoverage->at(i)) && "pCoverage NaN");
    }

    // etchant coverage
    for (size_t i = 0; i < numPoints; ++i) {
      if (pCoverage->at(i) < 1.) {
        if (etchantRate->at(i) == 0.) {
          eCoverage->at(i) = 0;
        } else {
          eCoverage->at(i) =
              (etchantRate->at(i) * totalEtchantFlux * (1 - pCoverage->at(i))) /
              (k_ie * ionEnhancedRate->at(i) * totalIonFlux + k_ev * F_ev +
               etchantRate->at(i) * totalEtchantFlux);
        }
      } else {
        eCoverage->at(i) = 0.;
      }
      assert(!std::isnan(eCoverage->at(i)) && "eCoverage NaN");
    }
  }

private:
  static constexpr double rho_SiO2 = 2.3 * 1e7; // in (1e15 atoms/cm³)
  static constexpr double rho_SiNx = 2.3 * 1e7; // in (1e15 atoms/cm³)
  static constexpr double rho_Si = 5.02 * 1e7;  // in (1e15 atoms/cm³)
  static constexpr double rho_p = 2 * 1e7;      // in (1e15 atoms/cm³)

  static constexpr double k_ie = 1.;
  static constexpr double k_ev = 1.;

  static constexpr double delta_p = 0.;

  static constexpr double kB = 0.000086173324; // m² kg s⁻² K⁻¹
  static constexpr double temperature = 300.;  // K

  // fluxes in (1e15 /cm²)
  const NumericType totalIonFlux;
  const NumericType totalEtchantFlux;
  const NumericType totalPolyFlux;
  const NumericType F_ev;

  const NumericType etchStopDepth = 0.;
};

// Parameters from:
// A. LaMagna and G. Garozzo "Factors affecting profile evolution in plasma
// etching of SiO2: Modeling and experimental verification" Journal of the
// Electrochemical Society 150(10) 2003 pp. 1896-1902

template <typename NumericType>
class FluorocarbonIon
    : public rayParticle<FluorocarbonIon<NumericType>, NumericType> {
public:
  FluorocarbonIon(const NumericType passedPower) : power(passedPower) {}
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
    assert(primID < localData.getVectorData(0).size() && "id out of bounds");
    assert(E >= 0 && "Negative energy ion");

    const auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 4 && "Error in calculating cos theta");

    const auto sqrtE = std::sqrt(E);
    const auto f_e_sp = (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;
    const auto Y_s = Ae_sp * std::max(sqrtE - sqrtE_th_sp, 0.) * f_e_sp;
    const auto Y_ie = Ae_ie * std::max(sqrtE - sqrtE_th_ie, 0.) * cosTheta;
    const auto Y_p = Ap_ie * std::max(sqrtE - sqrtE_th_p, 0.) * cosTheta;

    // sputtering yield Y_s
    localData.getVectorData(0)[primID] += rayWeight * Y_s;

    // ion enhanced etching yield Y_ie
    localData.getVectorData(1)[primID] += rayWeight * Y_ie;

    // polymer yield Y_p
    localData.getVectorData(2)[primID] += rayWeight * Y_p;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    const auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);
    const double Phi = std::acos(cosTheta);

    double Eref_peak;
    if (Phi >= Phi_inflect) {
      Eref_peak =
          1 - (1 - A) * std::pow((halfPI - Phi) / (halfPI - Phi_inflect), n_r);
    } else {
      Eref_peak = A * std::pow(Phi / Phi_inflect, n_l);
    }

    const double TempEnergy = Eref_peak * E;
    double NewEnergy;
    do {
      NewEnergy =
          TempEnergy + (std::min((E - TempEnergy), TempEnergy) + 0.05 * E) *
                           (1 - 2. * uniDist(Rng)) *
                           std::sqrt(std::fabs(std::log(uniDist(Rng))));
    } while (NewEnergy > E || NewEnergy < 0.);

    if (NewEnergy > 4.) {
      E = NewEnergy;
      auto direction = rayReflectionSpecular<NumericType>(rayDir, geomNormal);
      return std::pair<NumericType, rayTriple<NumericType>>{1 - Eref_peak,
                                                            direction};
    } else {
      return std::pair<NumericType, rayTriple<NumericType>>{
          1., rayTriple<NumericType>{0., 0., 0.}};
    }
  }
  void initNew(rayRNG &RNG) override final {
    do {
      auto rand1 = uniDist(RNG) * (twoPI - 2 * peak) + peak;
      E = (1 + std::cos(rand1)) * (power / 2 * 0.75 + 10);
    } while (E < 4.);
  }

  int getRequiredLocalDataSize() const override final { return 3; }
  NumericType getSourceDistributionPower() const override final { return 100.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"ionSputteringRate", "ionEnhancedRate",
                                    "ionpeRate"};
  }

  void logData(rayDataLog<NumericType> &dataLog) override final {
    NumericType max = 0.75 * power + 20 + 1e-6;
    int idx = static_cast<int>(50 * E / max);
    assert(idx < 50 && idx >= 0);
    dataLog.data[0][idx] += 1.;
  }

private:
  static constexpr double sqrtE_th_sp = 4.2426406871;
  static constexpr double sqrtE_th_ie = 2.;
  static constexpr double sqrtE_th_p = 2.;

  static constexpr double twoPI = 6.283185307179586;
  static constexpr double halfPI = 6.283185307179586 / 4;

  static constexpr double Ae_sp = 0.00339;
  static constexpr double Ae_ie = 0.0361;
  static constexpr double Ap_ie = 8 * 0.0361;

  static constexpr double B_sp = 9.3;

  static constexpr double Phi_inflect = 1.55334303;
  static constexpr double n_r = 1.;
  static constexpr double n_l = 10.;

  const double A = 1. / (1. + (n_l / n_r) * (halfPI / Phi_inflect - 1.));
  std::uniform_real_distribution<NumericType> uniDist;

  const NumericType power;
  static constexpr double peak = 0.2;
  NumericType E;
};

template <typename NumericType, int D>
class FluorocarbonPolymer
    : public rayParticle<FluorocarbonPolymer<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight * gamma_p;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    return std::pair<NumericType, rayTriple<NumericType>>{gamma_p, direction};
  }
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"polyRate"};
  }

private:
  static constexpr NumericType gamma_p = 0.26;
};

template <typename NumericType, int D>
class FluorocarbonEtchant
    : public rayParticle<FluorocarbonEtchant<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight * gamma_e;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    // const auto &phi_e = globalData.getVectorData(1)[id];
    NumericType sticking = gamma_e; // * phi_e;
    return std::pair<NumericType, rayTriple<NumericType>>{sticking, direction};
  }
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"etchantRate"};
  }

private:
  static constexpr NumericType gamma_e = 0.9;
};

template <typename NumericType, int D>
class FluorocarbonEtchantOnPoly
    : public rayParticle<FluorocarbonEtchantOnPoly<NumericType, D>,
                         NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit
    localData.getVectorData(0)[primID] += rayWeight * gamma_pe;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    // const auto &phi_pe = globalData.getVectorData(0)[id];
    NumericType sticking = gamma_pe; // * phi_pe;
    return std::pair<NumericType, rayTriple<NumericType>>{sticking, direction};
  }
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"etchantOnPolyRate"};
  }

private:
  static constexpr NumericType gamma_pe = 0.6;
};

template <typename NumericType, int D>
class FluorocarbonEtching : public psProcessModel<NumericType, D> {
public:
  FluorocarbonEtching(const double ionFlux, const double etchantFlux,
                      const double polyFlux, const NumericType rfBiasPower,
                      const NumericType etchStopDepth = 0.) {
    // particles
    auto ion = std::make_unique<FluorocarbonIon<NumericType>>(rfBiasPower);
    auto etchant = std::make_unique<FluorocarbonEtchant<NumericType, D>>();
    auto poly = std::make_unique<FluorocarbonPolymer<NumericType, D>>();
    auto etchantOnPoly =
        std::make_unique<FluorocarbonEtchantOnPoly<NumericType, D>>();

    // surface model
    auto surfModel =
        psSmartPointer<FluorocarbonSurfaceModel<NumericType, D>>::New(
            ionFlux, etchantFlux, polyFlux, etchStopDepth);

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();

    this->setSurfaceModel(surfModel);
    this->setVelocityField(velField);
    this->setProcessName("FluorocarbonEtching");
    this->insertNextParticleType(ion, 50);
    this->insertNextParticleType(etchant);
    this->insertNextParticleType(poly);
    this->insertNextParticleType(etchantOnPoly);
  }
};