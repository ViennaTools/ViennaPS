#pragma once

#include <cmath>

#include <psModelParameters.hpp>

#include <psLogger.hpp>
#include <psMaterials.hpp>
#include <psProcessModel.hpp>

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

namespace Cl2N2EtchingImplementation {
template <class T, int D> class SurfaceModel : public psSurfaceModel<T> {
  using psSurfaceModel<T>::coverages;
  static constexpr double eps = 1e-6;

  static constexpr double kB_over_m_Cl = 234.6938;

  const double flux_part;
  const double flux_ev;
  const T etchStop;

public:
  SurfaceModel(const double pFlux, const double pTemperature, const T pEtchStop)
      : flux_part(0.25 * psParameters::Cl::rho *
                  std::sqrt(3 * kB_over_m_Cl * pTemperature)),
        flux_ev(pFlux * std::exp(-0.5 / (8.617e-5 * pTemperature))),
        etchStop(pEtchStop) {
    std::cout << "flux_part: " << flux_part << std::endl;
    std::cout << "flux_ev: " << flux_ev << std::endl;
  }

  void initializeCoverages(unsigned numGeometryPoints) override {
    if (coverages == nullptr) {
      coverages = psSmartPointer<psPointData<T>>::New();
    } else {
      coverages->clear();
    }
    std::vector<T> cov(numGeometryPoints);
    coverages->insertNextScalarData(cov, "coverage");
  }

  psSmartPointer<std::vector<T>>
  calculateVelocities(psSmartPointer<psPointData<T>> rates,
                      const std::vector<std::array<T, 3>> &coordinates,
                      const std::vector<T> &materialIds) override {

    updateCoverages(rates, materialIds);
    const auto numPoints = rates->getScalarData(0)->size();
    std::vector<T> etchRate(numPoints, 0.);

    const auto flux = rates->getScalarData("flux");
    const auto coverage = coverages->getScalarData("coverage");

    bool stop = false;

    for (size_t i = 0; i < numPoints; ++i) {
      if (coordinates[i][D - 1] < etchStop) {
        stop = true;
        break;
      }

      if (psMaterialMap::isMaterial(materialIds[i], psMaterial::TiN)) {
        etchRate[i] =
            -flux_part * flux->at(i) * coverage->at(i) / psParameters::TiN::rho;
      }
    }

    if (stop) {
      std::fill(etchRate.begin(), etchRate.end(), 0.);
      psLogger::getInstance().addInfo("Etch stop depth reached.").print();
    }

    return psSmartPointer<std::vector<T>>::New(std::move(etchRate));
  }

  void updateCoverages(psSmartPointer<psPointData<T>> rates,
                       const std::vector<T> &materialIds) override {
    // update coverage based on flux
    const auto numPoints = rates->getScalarData(0)->size();
    const auto flux = rates->getScalarData("flux");
    auto coverage = coverages->getScalarData("coverage");

    for (size_t i = 0; i < numPoints; ++i) {
      if (flux->at(i) > eps) {
        coverage->at(i) =
            flux_part * flux->at(i) / (flux_part * flux->at(i) + flux_ev);
      } else {
        coverage->at(i) = 0.;
      }
    }
  }
};

template <class T, int D>
class Particle : public rayParticle<Particle<T, D>, T> {
public:
  Particle(const T pStickingCoeff) : stickingCoeff(pStickingCoeff) {}
  void surfaceCollision(T rayWeight, const rayTriple<T> &rayDir,
                        const rayTriple<T> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<T> &localData,
                        const rayTracingData<T> *globalData,
                        rayRNG &Rng) override final {
    // Rate is normalized by dividing with the local sticking coefficient
    localData.getVectorData(0)[primID] += rayWeight;
  }
  std::pair<T, rayTriple<T>>
  surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                    const rayTriple<T> &geomNormal, const unsigned int primID,
                    const int materialId, const rayTracingData<T> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<T, D>(geomNormal, Rng);
    return std::pair<T, rayTriple<T>>{stickingCoeff, direction};
  }
  T getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return {"flux"};
  }

private:
  const T stickingCoeff;
};
} // namespace Cl2N2EtchingImplementation

/// Model for etching TiN with Cl2 and N2.
template <class T, int D> class psCl2N2Etching : public psProcessModel<T, D> {
public:
  psCl2N2Etching(const T flux, const T temperature, const T stickingCoefficient,
                 const T etchStop = std::numeric_limits<T>::lowest()) {
    // particle
    auto particle =
        std::make_unique<Cl2N2EtchingImplementation::Particle<T, D>>(
            stickingCoefficient);

    // surface model
    auto surfaceModel =
        psSmartPointer<Cl2N2EtchingImplementation::SurfaceModel<T, D>>::New(
            flux, temperature, etchStop);

    // velocity field
    auto velField = psSmartPointer<psDefaultVelocityField<T>>::New(2);

    this->setSurfaceModel(surfaceModel);
    this->setVelocityField(velField);
    this->setProcessName("Cl2N2Etching");
    this->insertNextParticleType(particle);
  }
};