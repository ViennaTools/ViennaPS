#include <rayParticle.hpp>
#include <vcSampling.hpp>

using namespace viennacore;

template <class NumericType> struct UnivariateDistribution {
  std::vector<NumericType> pdf;
  std::vector<NumericType> support; // (x-values)
};

template <class NumericType> struct BivariateDistribution {
  std::vector<std::vector<NumericType>> pdf; // 2D grid of values
  std::vector<NumericType> support_x;        // (x-values)
  std::vector<NumericType> support_y;        // (y-values)
};

template <class NumericType, int D>
class UnivariateDistributionParticle final
    : public viennaray::Particle<UnivariateDistributionParticle<NumericType, D>,
                                 NumericType> {
  NumericType energy_;
  Sampling<NumericType, 1, false>
      directionSampling_; // uni-variate (1D) sampling instance
                          // using inverse-transform sampling
  Sampling<NumericType, 1, true>
      energySampling_; // uni-variate (1D) sampling instance
                       // using Alias sampling

public:
  UnivariateDistributionParticle(
      const UnivariateDistribution<NumericType> &directionDist,
      const UnivariateDistribution<NumericType> &energyDist) {
    // initialize sampling instances with custom distribution
    directionSampling_.setPDF(directionDist.pdf, directionDist.support);
    energySampling_.setPDF(energyDist.pdf, energyDist.support);
  }

  // This new initialize function can be overridden to generate a custom
  // direction for the particle
  Vec3D<NumericType> initNewWithDirection(RNG &rngState) override {

    ////// Custom Energy Sampling
    auto energySample = energySampling_.sample(rngState);
    // the returned sample is a 1D array
    energy_ = energySample[0];

    ////// Custom Direction Sampling
    Vec3D<NumericType> direction = {0., 0., 0.};
    auto directionSample = directionSampling_.sample(rngState);
    // the returned sample is a 1D array
    NumericType theta = directionSample[0];
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

    std::cout << "Energy is " << energy_ << std::endl;
    std::cout << "Direction is " << direction << std::endl;

    // THE RETURNED DIRECTION HAS TO BE NORMALIZED
    // if this returns a 0-vector the default direction from
    // the power cosine distribution is used
    return direction;
  }

  // override all other functions as usual
};

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

  // This new initialize function can be overridden to generate a custom
  // direction for the particle
  Vec3D<NumericType> initNewWithDirection(RNG &rngState) override {

    // Generate sample
    auto sample = directionNEnergySampling_.sample(rngState);
    // sample is 2D array

    NumericType theta = sample[0];
    energy_ = sample[1];

    Vec3D<NumericType> direction = {0., 0., 0.};
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

    std::cout << "Energy is " << energy_ << std::endl;
    std::cout << "Direction is " << direction << std::endl;

    // THE RETURNED DIRECTION HAS TO BE NORMALIZED
    // if this returns a 0-vector the default direction from
    // the power cosine distribution is used
    return direction;
  }

  // override all other functions as usual
};

int main() {
  using NumericType = double;
  constexpr int D = 2;
  Logger::setLogLevel(LogLevel::DEBUG);

  constexpr int N = 1000; // pdf resolution

  // ----- Setup with 2 uni-variate distributions -----
  {
    UnivariateDistribution<NumericType> angularDistribution;
    angularDistribution.pdf.resize(N);
    angularDistribution.support.resize(N);
    for (int i = 0; i < N; ++i) {
      const NumericType theta = i * M_PI_2 / N;
      angularDistribution.support[i] = theta;
      angularDistribution.pdf[i] =
          std::exp(-theta * theta / 0.5) * std::sin(theta);
    }

    constexpr NumericType maxEnergy = 200.;
    constexpr NumericType meanEnergy = 100.;
    UnivariateDistribution<NumericType> energyDistribution;
    energyDistribution.pdf.resize(N);
    energyDistribution.support.resize(N);
    for (int i = 0; i < N; ++i) {
      const NumericType energy = i * maxEnergy / N;
      energyDistribution.support[i] = energy;
      energyDistribution.pdf[i] =
          std::exp(-(energy - meanEnergy) * (energy - meanEnergy) / 20);
    }

    auto testParticle =
        std::make_unique<UnivariateDistributionParticle<NumericType, D>>(
            angularDistribution, energyDistribution);
    // check if particle can be copied correctly with sampling instances
    auto testCopy = testParticle->clone();

    RNG rngState(236521);
    for (int i = 0; i < 10; ++i) {
      auto direction = testCopy->initNewWithDirection(rngState);
      assert(IsNormalized(direction));
    }
  }

  // ----- Setup with 1 bi-variate distributions -----
  {
    BivariateDistribution<NumericType> angularEnergyDistribution;
    angularEnergyDistribution.support_x.resize(N);
    angularEnergyDistribution.support_y.resize(N);
    angularEnergyDistribution.pdf.resize(N);

    constexpr NumericType maxEnergy = 200.;
    constexpr NumericType meanEnergy = 100.;
    for (int i = 0; i < N; ++i) {
      const NumericType theta = i * M_PI_2 / N;
      NumericType energy = i * maxEnergy / N;
      angularEnergyDistribution.support_x[i] = theta;
      angularEnergyDistribution.support_y[i] = energy;

      angularEnergyDistribution.pdf[i].resize(N);
      for (int j = 0; j < N; ++j) {
        energy = j * maxEnergy / N;
        // some made-up PDF
        const NumericType pdfVal =
            std::exp(-theta * theta / 0.5) * std::sin(theta) *
            std::exp(-(energy - meanEnergy) * (energy - meanEnergy) / 20);
        angularEnergyDistribution.pdf[i][j] = pdfVal;
      }
    }

    auto testParticle =
        std::make_unique<BivariateDistributionParticle<NumericType, D>>(
            angularEnergyDistribution);
    auto testCopy = testParticle->clone();

    RNG rngState(235123612);
    for (int i = 0; i < 10; ++i) {
      auto direction = testCopy->initNewWithDirection(rngState);
      assert(IsNormalized(direction));
    }
  }
}