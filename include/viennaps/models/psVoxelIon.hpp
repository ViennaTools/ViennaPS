#pragma once

// An ion on a voxel grid.
//
// Everything else this arm traces is a neutral: it sticks with a probability
// and re-emits diffusely, and its rate law only ever sees a flux. An ion is
// different in the three ways that make plasma etching directional, and all
// three have to be carried, not approximated:
//
//   * it arrives with an ENERGY, drawn from the source distribution, and loses
//     some of it at every glancing reflection;
//   * what it does on arrival depends on the ANGLE OF INCIDENCE against the
//     local surface normal, through the yield Y = A max(sqrt(E)-sqrt(Eth),0)
//     f(theta);
//   * it is absorbed on a steep hit and reflects on a grazing one, into a cone
//     about the specular direction rather than a hemisphere.
//
// The physics is taken from ViennaPS's own ChemicalIon
// (psSurfaceChemistry.hpp) and its helpers in psIonModelUtil.hpp, and this
// class calls the same functions, so the two arms differ only in how a ray
// finds the surface and what normal it meets there.
//
// THE NORMAL IS THE POINT. Every mechanism run so far is thermal, and the
// normal enters only through a re-emission direction, so a face normal and a
// Youngs normal gave answers differing in the third decimal. An ion yield
// depends on cos(theta) explicitly, so the 15-60 degrees a face normal is
// wrong by on a tilted surface acts directly on the result. This is the case
// the two estimators exist to be compared on.

#include "psChemicalMechanismIO.hpp"
#include "psIonModelUtil.hpp"
#include "psSurfaceChemistry.hpp"

#include <csVoxelFlux.hpp>

#include <rayReflection.hpp>

namespace viennaps {

using namespace viennacore;

template <class NumericType, int D> class VoxelIonFlux {
  using IonSource = typename ChemicalMechanism<NumericType>::IonSource;
  using IonYield = typename ChemicalMechanism<NumericType>::IonYield;

  const ChemicalMechanism<NumericType> &mech_;
  const viennacs::LatticeMap<NumericType, D> &lattice_;
  const std::vector<NumericType> &fill_;
  const std::vector<int> &material_;
  viennacs::VoxelFlux<NumericType, D> walker_; ///< for traceToSurface/deposit
  viennacs::VoxelAdvance<NumericType, D> areas_;

  NumericType aEnergy_ = 0;
  NumericType minEth_ = 0;

public:
  VoxelIonFlux(const ChemicalMechanism<NumericType> &mech,
               const viennacs::LatticeMap<NumericType, D> &lattice,
               const std::vector<NumericType> &fill,
               const std::vector<int> &material,
               viennacs::NormalEstimator estimator =
                   viennacs::NormalEstimator::FillGradientYoungs)
      : mech_(mech), lattice_(lattice), fill_(fill), material_(material),
        walker_(lattice, fill, estimator), areas_(lattice) {
    const auto &src = mech_.ionSource;
    aEnergy_ = NumericType(1) /
               (NumericType(1) +
                src.n_l * (NumericType(M_PI_2) /
                               (src.inflectAngle * NumericType(M_PI) / 180) -
                           NumericType(1)));
    minEth_ = std::numeric_limits<NumericType>::max();
    for (const auto &y : mech_.ionYields)
      minEth_ = std::min(minEth_, y.Eth);
    if (mech_.ionYields.empty())
      minEth_ = 0;
  }

  /// Yield-weighted flux per channel, per cell. Index c follows
  /// mech.ionYields, whose gasIndex names the species each channel writes.
  void setTraversalEngine(viennacs::TraversalEngine e) {
    walker_.setTraversalEngine(e);
  }

  std::vector<std::vector<NumericType>> trace(size_t numRays,
                                              unsigned seed = 1) const {
    walker_.prepareTransport(); // the walker's rays fly outside its trace()
    const size_t nChannels = mech_.ionYields.size();
    std::vector<std::vector<NumericType>> flux(
        nChannels, std::vector<NumericType>(fill_.size(), NumericType(0)));
    if (nChannels == 0)
      return flux;

    const auto &src = mech_.ionSource;
    const NumericType delta = lattice_.gridDelta();
    const auto &dims = lattice_.dims();
    const auto &minCorner = lattice_.minCorner();

    std::array<NumericType, D> sourceMin{}, sourceMax{};
    for (int d = 0; d < D; ++d) {
      sourceMin[d] = minCorner[d];
      sourceMax[d] = minCorner[d] + delta * static_cast<NumericType>(dims[d]);
    }
    NumericType sourceArea = 1;
    for (int d = 0; d < D - 1; ++d)
      sourceArea *= (sourceMax[d] - sourceMin[d]);

    // The ion's source is far more directional than a neutral's: an exponent
    // of several hundred is a near-collimated beam, which is what makes the
    // sidewalls of a feature see almost no ion flux.
    const NumericType exponent = NumericType(1) / (src.exponent + NumericType(1));
    const NumericType rayRate =
        mech_.gas[mech_.ionYields[0].gasIndex].sourceFlux * sourceArea /
        static_cast<NumericType>(numRays);

    const NumericType thetaRMin = src.thetaRMin * NumericType(M_PI) / 180;
    const NumericType thetaRMax = src.thetaRMax * NumericType(M_PI) / 180;
    const NumericType inflect = src.inflectAngle * NumericType(M_PI) / 180;
    const NumericType minAngle = src.minAngle * NumericType(M_PI) / 180;

    std::vector<std::vector<NumericType>> collected(
        nChannels, std::vector<NumericType>(fill_.size(), NumericType(0)));

#pragma omp parallel
    {
      std::vector<std::vector<NumericType>> mine(
          nChannels, std::vector<NumericType>(fill_.size(), NumericType(0)));
      const unsigned thread =
#ifdef _OPENMP
          static_cast<unsigned>(omp_get_thread_num());
#else
          0u;
#endif
      RNG rng(seed * 7919u + thread);
      std::uniform_real_distribution<NumericType> uni(NumericType(0),
                                                      NumericType(1));

#pragma omp for schedule(static)
      for (long long r = 0; r < static_cast<long long>(numRays); ++r) {
        std::array<NumericType, D> origin{}, direction{};
        for (int d = 0; d < D - 1; ++d)
          origin[d] = sourceMin[d] + (sourceMax[d] - sourceMin[d]) * uni(rng);
        origin[D - 1] = sourceMax[D - 1];

        // Same 3D-sample-and-project as the neutral source: the polar angle of
        // the 3D cosine law is not the angle of the 2D one.
        const NumericType cosT = std::pow(uni(rng), exponent);
        const NumericType sinT =
            std::sqrt(std::max(NumericType(0), NumericType(1) - cosT * cosT));
        const NumericType phi = NumericType(2) * NumericType(M_PI) * uni(rng);
        if constexpr (D == 2) {
          const NumericType dx = std::cos(phi) * sinT, dy = -cosT;
          const NumericType n = std::sqrt(dx * dx + dy * dy);
          direction[0] = dx / n;
          direction[1] = dy / n;
        } else {
          direction[0] = std::cos(phi) * sinT;
          direction[1] = std::sin(phi) * sinT;
          direction[2] = -cosT;
        }

        NumericType energy = impl::initNormalDistEnergy<NumericType>(
            rng, src.meanEnergy, src.sigmaEnergy);
        NumericType weight = rayRate;

        for (int bounce = 0; bounce < 200; ++bounce) {
          const auto hit = walker_.traceToSurface(origin, direction, rng);
          if (!hit.hit())
            break;

          NumericType dot = 0;
          for (int d = 0; d < D; ++d)
            dot += direction[d] * hit.normal[d];
          const NumericType cosTheta =
              std::min(NumericType(1), std::max(NumericType(0), -dot));
          const NumericType incAngle = std::acos(cosTheta);
          const NumericType sqrtE = std::sqrt(energy);
          const auto mat = MaterialMap::mapToMaterial(material_[hit.cellId]);

          for (size_t c = 0; c < nChannels; ++c) {
            const auto &y = mech_.ionYields[c];
            const NumericType A = y.materialA.get(mat);
            const NumericType Eth = y.materialEth.get(mat);
            NumericType f;
            if (y.enhanced) {
              f = cosTheta < NumericType(0.5)
                      ? std::max(NumericType(3) - NumericType(6) * incAngle /
                                                      NumericType(M_PI),
                                 NumericType(0))
                      : NumericType(1);
            } else {
              f = std::max((NumericType(1) +
                            y.B * (NumericType(1) - cosTheta * cosTheta)) *
                               cosTheta,
                           NumericType(0));
            }
            const NumericType Y =
                A * std::max(sqrtE - std::sqrt(Eth), NumericType(0)) * f;
            if (Y > NumericType(0))
              walker_.deposit(mine[c], hit.index, Y * weight);
          }

          // A steep hit is absorbed; a grazing one reflects.
          NumericType sticking = 1;
          if (incAngle > thetaRMin)
            sticking = NumericType(1) -
                       std::min(NumericType(1),
                                std::max(NumericType(0),
                                         (incAngle - thetaRMin) /
                                             (thetaRMax - thetaRMin)));
          if (sticking >= NumericType(1))
            break;

          const NumericType newEnergy = impl::updateEnergy<NumericType>(
              rng, energy, incAngle, aEnergy_, inflect, src.n_l);
          if (newEnergy <= minEth_)
            break;
          energy = newEnergy;

          Vec3D<NumericType> rayDir3{0, 0, 0}, normal3{0, 0, 0};
          for (int d = 0; d < D; ++d) {
            rayDir3[d] = direction[d];
            normal3[d] = hit.normal[d];
          }
          const auto newDir = viennaray::ReflectionConedCosine<NumericType, D>(
              rayDir3, normal3, rng,
              NumericType(M_PI_2) - std::min(incAngle, minAngle));

          weight *= (NumericType(1) - sticking);
          if (weight <= rayRate * NumericType(1e-4))
            break;

          // Restart clear of the interface, as the neutral tracer does.
          int axis = 0;
          NumericType steepest = 0;
          for (int d = 0; d < D; ++d)
            if (std::abs(normal3[d]) > steepest) {
              steepest = std::abs(normal3[d]);
              axis = d;
            }
          const int outward = normal3[axis] > 0 ? 1 : -1;
          int clear = 1;
          auto probe = hit.index;
          for (int stepOut = 0; stepOut < 8; ++stepOut) {
            probe[axis] += outward;
            const int nid = lattice_.cellId(probe);
            if (nid < 0 || fill_[nid] <= NumericType(1e-9))
              break;
            ++clear;
          }
          for (int d = 0; d < D; ++d) {
            origin[d] = hit.point[d] +
                        normal3[d] * delta *
                            (static_cast<NumericType>(clear) + NumericType(1e-3));
            direction[d] = newDir[d];
          }
        }
      }

#pragma omp critical
      for (size_t c = 0; c < nChannels; ++c)
        for (size_t i = 0; i < mine[c].size(); ++i)
          collected[c][i] += mine[c][i];
    }

    // Rates become flux densities, and are smoothed, exactly as the neutral
    // channels are.
    NumericType faceArea = 1;
    for (int d = 0; d < D - 1; ++d)
      faceArea *= delta;
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);

    std::array<int, D> idx{};
    for (size_t flat = 0; flat < sites; ++flat) {
      size_t rem = flat;
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lattice_.cellId(idx);
      if (id < 0)
        continue;
      const NumericType area = walker_.areaAt(idx); // per-trace cache
      if (area <= NumericType(1e-2) * faceArea)
        continue;
      for (size_t c = 0; c < nChannels; ++c)
        flux[c][id] = collected[c][id] / area;
    }
    for (size_t c = 0; c < nChannels; ++c)
      walker_.smooth(flux[c], 1);
    return flux;
  }
};

} // namespace viennaps
