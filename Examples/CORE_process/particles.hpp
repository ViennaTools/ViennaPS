#ifndef PARTICLES_HPP
#define PARTICLES_HPP

#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayUtil.hpp>

// Parameters from:
// A. LaMagna and G. Garozzo "Factors affecting profile evolution in plasma
// etching of SiO2 modelling and experimental verification" Journal of the
// Electrochemical Society 150(10) 2003 pp. 1896-1902

template <typename NumericType>
class Ion : public rayParticle<Ion<NumericType>, NumericType>
{
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final
  {
    // collect data for this hit
    assert(primID < localData.getVectorData(0).size() && "id out of bounds");
    assert(E >= 0 && "Negative energy ion");

    const auto cosTheta = -rayInternal::DotProduct(rayDir, geomNormal);

    assert(cosTheta >= 0 && "Hit backside of disc");
    assert(cosTheta <= 1 + 1e6 && "Error in calculating cos theta");

    const auto sqrtE = std::sqrt(E);
    double f_ie;
    if (cosTheta > 0.5)
    {
      f_ie = 1.;
    }
    else
    {
      f_ie = std::max(3. - 6. * std::acos(cosTheta) / rayInternal::PI, 0.);
    }
    const auto Y_s = Ae_sp * std::max(sqrtE - sqrtE_th_sp, 0.);
    const auto Y_ie = Ae_ie * std::max(sqrtE - sqrtE_th_ie, 0.) * f_ie;

    // sputtering yield Y_s ionSputteringRate
    localData.getVectorData(0)[primID] += rayWeight * Y_s;

    // ion enhanced etching yield Y_ie ionEnhancedRate
    localData.getVectorData(1)[primID] += rayWeight * Y_ie;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final
  {
    // TODO
    return std::pair<NumericType, rayTriple<NumericType>>{
        1., rayTriple<NumericType>{0., 0., 0.}};
  }
  void initNew(rayRNG &RNG) override final
  {
    std::uniform_real_distribution<NumericType> uniDist;
    do
    {
      const auto rand1 = uniDist(RNG);
      const auto rand2 = uniDist(RNG);
      E = std::cos(twoPI * rand1) * std::sqrt(-2. * std::log(rand2)) *
              deltaEnergy +
          meanEnergy;
    } while (E < 0);
  }

  int getRequiredLocalDataSize() const override final { return 2; }
  NumericType getSourceDistributionPower() const override final { return 80.; }
  std::vector<std::string> getLocalDataLabels() const override final
  {
    return std::vector<std::string>{"ionSputteringRate", "ionEnhancedRate"};
  }

private:
  static constexpr double sqrtE_th_sp = 4.47213595499958;
  static constexpr double sqrtE_th_ie = 3.872983346207417;

  static constexpr double meanEnergy = 100;
  static constexpr double deltaEnergy = 40;
  static constexpr double twoPI = 6.283185307179586;

  static constexpr double Ae_sp = 0.0337;
  static constexpr double Ae_ie = 7.;

  NumericType E;
};

template <typename NumericType, int D>
class Etchant : public rayParticle<Etchant<NumericType, D>, NumericType>
{
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final
  {
    if (materialId == 2)
      localData.getVectorData(0)[primID] += rayWeight * gamma_e_p;
    else if (materialId == 1)
      localData.getVectorData(0)[primID] += rayWeight * gamma_e_Si;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final
  {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    NumericType sticking;
    if (materialId == 0)
      sticking = 0;
    else if (materialId == 1)
      sticking = gamma_e_Si;
    else
      sticking = gamma_e_p;

    return std::pair<NumericType, rayTriple<NumericType>>{sticking, direction};
  }
  void initNew(rayRNG &RNG) override final {}
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final
  {
    return std::vector<std::string>{"etchantRate"};
  }

private:
  static constexpr NumericType gamma_e_Si = 0.7;
  static constexpr NumericType gamma_e_p = 0.6;
};

#endif