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
class Ion : public rayParticle<Ion<NumericType>, NumericType> {
public:
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
    assert(cosTheta <= 1 + 1e6 && "Error in calculating cos theta");

    const auto sqrtE = std::sqrt(E);
    const auto f_e_sp = (1 + B_sp * (1 - cosTheta * cosTheta)) * cosTheta;
    const auto Y_s = Ae_sp * std::max(sqrtE - sqrtE_th_sp, 0.) * f_e_sp;
    const auto Y_ie = Ae_ie * std::max(sqrtE - sqrtE_th_ie, 0.) * cosTheta;
    const auto Y_p = Ap_ie * std::max(sqrtE - sqrtE_th_p, 0.) * cosTheta;

    // sputtering yield Y_s ionSputteringRate
    localData.getVectorData(0)[primID] += rayWeight * Y_s;

    // ion enhanced etching yield Y_ie ionEnhancedRate
    localData.getVectorData(1)[primID] += rayWeight * Y_ie;

    // polymer yield Y_p ionpeRate
    localData.getVectorData(2)[primID] += rayWeight * Y_p;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    return std::pair<NumericType, rayTriple<NumericType>>{
        1., rayTriple<NumericType>{0., 0., 0.}};
  }
  void initNew(rayRNG &RNG) override final {
    std::uniform_real_distribution<NumericType> uniDist;
    do {
      const auto rand1 = uniDist(RNG);
      const auto rand2 = uniDist(RNG);
      E = std::cos(twoPI * rand1) * std::sqrt(-2. * std::log(rand2)) *
              deltaEnergy +
          meanEnergy;
    } while (E < 0);
  }

  int getRequiredLocalDataSize() const override final { return 3; }
  NumericType getSourceDistributionPower() const override final { return 80.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"ionSputteringRate", "ionEnhancedRate",
                                    "ionpeRate"};
  }

private:
  static constexpr double sqrtE_th_sp = 4.2426406871;
  static constexpr double sqrtE_th_ie = 2.;
  static constexpr double sqrtE_th_p = 2.;

  static constexpr double meanEnergy = 70;
  static constexpr double deltaEnergy = 30;
  static constexpr double twoPI = 6.283185307179586;

  static constexpr double Ae_sp = 0.00339;
  static constexpr double Ae_ie = 0.0361;
  static constexpr double Ap_ie = 8 * 0.0361;

  static constexpr double B_sp = 9.3;
  NumericType E;
};

template <typename NumericType, int D>
class Polymer : public rayParticle<Polymer<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit

    // this is bad
    // unsigned int id = primID;
    // if (id >= globalData.getVectorData(0).size())
    // {
    //   // std::cout << "ERROR: primitive ID is out of range of coverages!" <<
    //   std::endl; id = globalData.getVectorData(0).size() - 1;
    // }

    // const auto phi_pe = globalData.getVectorData(0)[id];
    // const auto phi_p = globalData.getVectorData(1)[id];
    // const auto phi_e = globalData.getVectorData(2)[id];

    const auto Sp = gamma_p; // * std::max(1. - phi_e - phi_p * phi_pe, 0.);
    localData.getVectorData(0)[primID] += rayWeight * Sp;
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
  void initNew(rayRNG &RNG) override final {}
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"polyRate"};
  }

private:
  static constexpr NumericType gamma_p = 0.26;
};

template <typename NumericType, int D>
class Etchant : public rayParticle<Etchant<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit

    // this is bad
    // unsigned int id = primID;
    // if (id >= globalData.getVectorData(0).size())
    // {
    //   // std::cout << "ERROR: primitive ID is out of range of coverages!" <<
    //   std::endl; id = globalData.getVectorData(0).size() - 1;
    // }
    // while (id >= globalData.getVectorData(0).size())
    // {
    //   // std::cout << "ERROR: primitive ID is out of range of coverages!" <<
    //   std::endl; id = globalData.getVectorData(0).size() - 1;
    // }

    // const auto &phi_e = globalData.getVectorData(0)[id];
    const auto Se = gamma_e; // * std::max(1. - phi_e, 0.);

    // etchanteRate
    localData.getVectorData(0)[primID] += rayWeight * Se;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    return std::pair<NumericType, rayTriple<NumericType>>{gamma_e, direction};
  }
  void initNew(rayRNG &RNG) override final {}
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"etchanteRate"};
  }

private:
  static constexpr NumericType gamma_e = 0.9;
};

template <typename NumericType, int D>
class EtchantPoly
    : public rayParticle<EtchantPoly<NumericType, D>, NumericType> {
public:
  void surfaceCollision(NumericType rayWeight,
                        const rayTriple<NumericType> &rayDir,
                        const rayTriple<NumericType> &geomNormal,
                        const unsigned int primID, const int materialId,
                        rayTracingData<NumericType> &localData,
                        const rayTracingData<NumericType> *globalData,
                        rayRNG &Rng) override final {
    // collect data for this hit

    // this is bad
    // unsigned int id = primID;
    // if (id >= globalData.getVectorData(0).size())
    // {
    //   // std::cout << "ERROR: primitive ID is out of range of coverages!" <<
    //   std::endl; id = globalData.getVectorData(0).size() - 1;
    // }

    // const auto &phi_pe = globalData.getVectorData(0)[id];
    const auto Spe = gamma_pe; // * std::max(1. - phi_pe, 0.);
    localData.getVectorData(0)[primID] += rayWeight * Spe;
  }
  std::pair<NumericType, rayTriple<NumericType>>
  surfaceReflection(NumericType rayWeight, const rayTriple<NumericType> &rayDir,
                    const rayTriple<NumericType> &geomNormal,
                    const unsigned int primId, const int materialId,
                    const rayTracingData<NumericType> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<NumericType, D>(geomNormal, Rng);
    return std::pair<NumericType, rayTriple<NumericType>>{gamma_pe, direction};
  }
  void initNew(rayRNG &RNG) override final {}
  int getRequiredLocalDataSize() const override final { return 1; }
  NumericType getSourceDistributionPower() const override final { return 1.; }
  std::vector<std::string> getLocalDataLabels() const override final {
    return std::vector<std::string>{"etchantpeRate"};
  }

private:
  static constexpr NumericType gamma_pe = 0.6;
};

#endif