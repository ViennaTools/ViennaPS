#include <geometries/psMakeTrench.hpp>
#include <psConstants.hpp>
#include <psProcess.hpp>
#include <rayParticle.hpp>
#include <vcSampling.hpp>

#include "loadDistribution.hpp"

using namespace viennaps;

template <class NumericType, int D>
class BivariateDistributionParticle final
    : public viennaray::Particle<BivariateDistributionParticle<NumericType, D>,
                                 NumericType> {
  NumericType energy_;
  Sampling<NumericType, 2>
      directionNEnergySampling_; // bivariate (2D) sampling instance using
                                 // accept-reject sampling

public:
  explicit BivariateDistributionParticle(
      const BivariateDistribution<NumericType> &angleEnergyDistribution) {
    // initialize sampling instances with custom distribution
    directionNEnergySampling_.setPDF(angleEnergyDistribution.pdf,
                                     angleEnergyDistribution.support_x,
                                     angleEnergyDistribution.support_y);
  }

  Vec3D<NumericType> initNewWithDirection(RNG &rngState) override {

    // Generate sample
    auto sample = directionNEnergySampling_.sample(rngState);

    assert(sample[1] >= -90. && sample[1] <= 90.);
    NumericType theta = constants::degToRad(sample[1]);
    energy_ = sample[0];

    Vec3D<NumericType> direction{0., 0., 0.};
    if constexpr (D == 2) {
      direction[0] = std::sin(theta);
      direction[1] = -std::cos(theta);
    } else {
      std::uniform_real_distribution<NumericType> uniform_dist(0, 2 * M_PI);
      auto phi = uniform_dist(rngState);
      direction[0] = std::sin(theta) * std::cos(phi);
      direction[1] = std::sin(theta) * std::sin(phi);
      direction[2] = -std::cos(theta);
    }

    return direction;
  }

  void surfaceCollision(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                        const Vec3D<NumericType> &geomNormal,
                        const unsigned int primID, const int,
                        viennaray::TracingData<NumericType> &localData,
                        const viennaray::TracingData<NumericType> *,
                        RNG &) override {
    localData.getVectorData(0)[primID] +=
        std::max(std::sqrt(energy_) - std::sqrt(thresholdEnergy_), 0.);
  }

  std::pair<NumericType, Vec3D<NumericType>>
  surfaceReflection(NumericType rayWeight, const Vec3D<NumericType> &rayDir,
                    const Vec3D<NumericType> &geomNormal,
                    const unsigned int primID, const int materialId,
                    const viennaray::TracingData<NumericType> *globalData,
                    RNG &rngState) override {

    NumericType incAngle = std::acos(-DotProduct(rayDir, geomNormal));
    NumericType Eref_peak;
    NumericType A_ = (1. / (1. + n_l_ * (M_PI_2 / inflectAngle_ - 1.)));
    if (incAngle >= inflectAngle_) {
      Eref_peak =
          1. - (1. - A_) * (M_PI_2 - incAngle) / (M_PI_2 - inflectAngle_);
    } else {
      Eref_peak = A_ * std::pow(incAngle / inflectAngle_, n_l_);
    }

    std::normal_distribution<NumericType> normalDist(Eref_peak * energy_,
                                                     0.1 * energy_);
    NumericType newEnergy;
    do {
      newEnergy = normalDist(rngState);
    } while (newEnergy > energy_ || newEnergy < 0.);

    if (newEnergy > thresholdEnergy_) {
      energy_ = newEnergy;
      auto direction = viennaray::ReflectionConedCosine<NumericType, D>(
          rayDir, geomNormal, rngState, M_PI_2 - std::min(incAngle, minAngle_));
      return std::pair<NumericType, Vec3D<NumericType>>{0., direction};
    } else {
      return std::pair<NumericType, Vec3D<NumericType>>{
          1., Vec3D<NumericType>{0., 0., 0.}};
    }
  }

  std::vector<std::string> getLocalDataLabels() const override {
    return {"particleFlux"};
  }

private:
  const NumericType thresholdEnergy_ = 10.0;
  const NumericType inflectAngle_ = constants::degToRad(89);
  const NumericType minAngle_ = constants::degToRad(5.);
  const NumericType n_l_ = 10;
};

template <typename NumericType>
class CustomSurfaceModel : public SurfaceModel<NumericType> {
public:
  SmartPointer<std::vector<NumericType>>
  calculateVelocities(SmartPointer<viennals::PointData<NumericType>> fluxes,
                      const std::vector<Vec3D<NumericType>> &coordinates,
                      const std::vector<NumericType> &materialIds) override {
    auto velocity =
        SmartPointer<std::vector<NumericType>>::New(materialIds.size(), 0.);
    auto flux = fluxes->getScalarData("particleFlux");

    for (std::size_t i = 0; i < velocity->size(); i++) {
      if (!MaterialMap::isMaterial(materialIds[i], Material::Mask)) {
        velocity->at(i) = -flux->at(i);
      }
    }

    return velocity;
  }
};

int main() {

  Logger::setLogLevel(LogLevel::DEBUG);
  using NumericType = double;
  constexpr int D = 2;

  const NumericType gridDelta = 0.1;
  const NumericType xExtent = 10.0;
  auto domain = Domain<NumericType, D>::New(gridDelta, xExtent,
                                            BoundaryType::REFLECTIVE_BOUNDARY);
  MakeTrench<NumericType, D>(domain, 4.0, 0.0, 0.0, 1.0).apply();

  auto model = SmartPointer<ProcessModel<NumericType, D>>::New();
  model->setProcessName("CustomParticleDistribution");
  auto surfaceModel = SmartPointer<CustomSurfaceModel<NumericType>>::New();
  model->setSurfaceModel(surfaceModel);
  auto velField = SmartPointer<DefaultVelocityField<NumericType, D>>::New(2);
  model->setVelocityField(velField);

  auto distribution =
      loadDistributionFromFile<NumericType>("custom_distribution.txt");

  auto particle =
      std::make_unique<BivariateDistributionParticle<NumericType, D>>(
          distribution);
  model->insertNextParticleType(particle);

  Process<NumericType, D>(domain, model, 2.0).apply();

  domain->saveVolumeMesh("customParticleDistribution");

  return 0;
}