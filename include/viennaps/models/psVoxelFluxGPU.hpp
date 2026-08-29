#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include <csVoxelAdvance.hpp>
#include <csVoxelFlux.hpp>
#include <csVoxelInteraction.hpp>

// for SurfaceChemistryParamsGPU: the ion runs on the SAME device callables
// the level-set GPU arm uses, so it is parameterised by the same struct
#include "psSurfaceChemistry.hpp"

#include <gpu/raygTraceCell.hpp>

#include <vcContext.hpp>
#include <vcCudaBuffer.hpp>

#include <array>
#include <memory>
#include <vector>

namespace viennaps {

using namespace viennacore;

/// The voxel neutral transport on the GPU: the surface band as OptiX cell
/// primitives, the per-cell sticking the chemistry computed on the host
/// uploaded beside them, and ViennaRay's cell pipeline doing what the CPU
/// engines do -- deposit the full incident weight, re-emit (1-s) diffusely
/// about the Youngs normal.
///
/// The DOWNLOADED result is raw incident rate per band cell. The CPU
/// convention -- spread over the interface neighbourhood, divide by area,
/// smooth one ring -- is applied on the host with the CPU flux's own
/// operators, which is exactly equivalent: all three are linear in the
/// per-cell totals, and the per-encounter spreading the CPU does commutes
/// with summation over encounters.
template <class NumericType, int D> class VoxelFluxGPU {
  const viennacs::LatticeMap<NumericType, D> *lattice_;
  const std::vector<NumericType> *fill_;
  viennacs::VoxelFlux<NumericType, D> ops_; ///< host-side spread/area/smooth
  viennacs::VoxelAdvance<NumericType, D> advance_;

  std::shared_ptr<DeviceContext> context_;
  viennaray::gpu::TraceCell<float, D> tracer_;
  CudaBuffer stickingBuffer_;
  std::vector<int> primCell_; ///< primID -> cell id
  std::vector<std::array<int, D>> primIdx_;
  std::vector<float> walkOut_; ///< per band cell, the restart displacement
  NumericType sourceArea_ = 0;

  // The ion: its own tracer, because its particle, callables, and result
  // channels differ, while the geometry is shared per step.
  viennaray::gpu::TraceCell<float, D> ionTracer_;
  CudaBuffer ionParamsBuffer_;
  bool ionConfigured_ = false;
  size_t nChannels_ = 0;
  /// Source flux of the gas species each yield channel writes into. Applied
  /// per channel, as the level-set arm does; one shared flux would misscale
  /// every channel but the first.
  std::vector<NumericType> channelFlux_;

public:
  VoxelFluxGPU(const viennacs::LatticeMap<NumericType, D> &lattice,
               const std::vector<NumericType> &fill,
               std::shared_ptr<DeviceContext> context)
      : lattice_(&lattice), fill_(&fill), ops_(lattice, fill),
        advance_(lattice), context_(context), tracer_(context),
        ionTracer_(context) {
    viennaray::gpu::Particle<float> particle;
    particle.name = "VoxelNeutral";
    particle.dataLabels = {"flux"};
    std::unordered_map<std::string, unsigned int> pMap = {{"VoxelNeutral", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__particleCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__particleReflectionCellSticking"},
    };
    tracer_.setCallables("ViennaRayCallableWrapper", context_->modulePath);
    tracer_.setParticleCallableMap({pMap, cMap});
    tracer_.insertNextParticle(particle);
    // ONCE. This compiles the OptiX modules and builds the pipeline, which
    // depend on the particles and callables, not on the geometry; apply()
    // regenerates the shader binding table itself every launch. Calling it
    // per trace recompiles the pipeline for every species on every step,
    // leaks the previous modules, pipelines and streams, and charges the
    // JIT to the transport time the benchmark is trying to measure.
    tracer_.prepareParticlePrograms();
  }

  /// Sets the ion up once, from the mechanism: the SAME device callables and
  /// the same parameter struct the level-set GPU arm uses -- energy sampled
  /// on init, yield channels deposited per hit, energy-dependent coned
  /// reflection. A mechanism without an ion leaves this a no-op.
  template <class Mechanism> void configureIon(const Mechanism &mech) {
    if (ionConfigured_ || !mech.ionSource.present || mech.ionYields.empty())
      return;
    const auto &src = mech.ionSource;
    constexpr NumericType toRad = M_PI / 180.;

    SurfaceChemistryParamsGPU p{};
    p.meanEnergy = static_cast<float>(src.meanEnergy);
    p.sigmaEnergy = static_cast<float>(src.sigmaEnergy);
    p.inflectAngle = static_cast<float>(src.inflectAngle * toRad);
    p.n_l = static_cast<float>(src.n_l);
    p.minAngle = static_cast<float>(src.minAngle * toRad);
    p.thetaRMin = static_cast<float>(src.thetaRMin * toRad);
    p.thetaRMax = static_cast<float>(src.thetaRMax * toRad);

    viennaray::gpu::Particle<float> ion;
    ion.name = "VoxelIon";
    ion.sticking = 0.f;
    ion.cosineExponent = static_cast<float>(src.exponent);

    NumericType minEth = std::numeric_limits<NumericType>::max();
    int nYield = 0;
    for (const auto &y : mech.ionYields) {
      if (nYield >= SurfaceChemistryParamsGPU::maxYields)
        break;
      ion.dataLabels.push_back(y.label);
      channelFlux_.push_back(mech.gas[y.gasIndex].sourceFlux);
      p.yieldA[nYield] = static_cast<float>(y.A);
      p.yieldEth[nYield] = static_cast<float>(y.Eth);
      p.yieldB[nYield] = static_cast<float>(y.B);
      p.yieldEnhanced[nYield] = y.enhanced ? 1 : 0;
      minEth = std::min(minEth, y.Eth);

      int nOverride = 0;
#define PS_VOXEL_GPU_YIELD(id, sym, cat, dens, cond, color)                    \
  if (y.materialA.has(BuiltInMaterial::sym) ||                                 \
      y.materialEth.has(BuiltInMaterial::sym)) {                               \
    if (nOverride < SurfaceChemistryParamsGPU::maxMaterials) {                 \
      const auto mat = Material(BuiltInMaterial::sym);                         \
      p.yieldOverrideMaterial[nYield][nOverride] = mat.legacyId();             \
      p.yieldOverrideA[nYield][nOverride] =                                    \
          static_cast<float>(y.materialA.get(mat));                            \
      p.yieldOverrideEth[nYield][nOverride] =                                  \
          static_cast<float>(y.materialEth.get(mat));                          \
      ++nOverride;                                                             \
    }                                                                          \
  }
      BUILTIN_MATERIAL_LIST(PS_VOXEL_GPU_YIELD)
#undef PS_VOXEL_GPU_YIELD
      p.yieldNumOverrides[nYield] = nOverride;
      ++nYield;
    }
    p.numYields = nYield;
    p.minEth = static_cast<float>(minEth);
    nChannels_ = static_cast<size_t>(nYield);

    ionParamsBuffer_.alloc(sizeof(p));
    ionParamsBuffer_.upload(&p, 1);

    std::unordered_map<std::string, unsigned int> pMap = {{"VoxelIon", 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__chemicalIonCollision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__chemicalIonReflection"},
        {0, viennaray::gpu::CallableSlot::INIT,
         "__direct_callable__chemicalIonInit"},
    };
    ionTracer_.setCallables("ViennaPSCallableWrapper", context_->modulePath);
    ionTracer_.setParticleCallableMap({pMap, cMap});
    ionTracer_.insertNextParticle(ion);
    ionTracer_.setParameters(ionParamsBuffer_.dPointer());
    ionTracer_.prepareParticlePrograms(); // once, as above
    ionConfigured_ = true;
  }

  ~VoxelFluxGPU() {
    stickingBuffer_.free();
    ionParamsBuffer_.free();
  }

  // holds device buffers and a tracer bound to them: not copyable or movable
  VoxelFluxGPU(const VoxelFluxGPU &) = delete;
  VoxelFluxGPU &operator=(const VoxelFluxGPU &) = delete;

  bool ionConfigured() const { return ionConfigured_; }

  /// Rebuilds the device geometry from the CURRENT fills: the band, the
  /// boxes, the Youngs normals. Once per step, before the species traces --
  /// fills do not move between them. `effectiveMaterial`, when given, also
  /// arms the ion tracer: the yield overrides resolve per material, so the
  /// ion must know what every band cell is made of.
  void prepareGeometry(viennacs::NormalEstimator estimator,
                       const std::vector<int> *effectiveMaterial = nullptr) {
    const auto band = advance_.interfaceBand(*fill_);
    viennacs::VoxelInteraction<NumericType, D> interaction(*lattice_, *fill_,
                                                           estimator);
    const auto &dims = lattice_->dims();
    const auto &minC = lattice_->minCorner();
    const NumericType delta = lattice_->gridDelta();
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);

    viennaray::gpu::CellGrid grid;
    grid.gridDelta = static_cast<float>(delta);
    for (int d = 0; d < 3; ++d) {
      grid.minimumExtent[d] = 0.f;
      grid.maximumExtent[d] = 0.f;
    }
    for (int d = 0; d < D; ++d) {
      grid.minimumExtent[d] = static_cast<float>(minC[d]);
      grid.maximumExtent[d] =
          static_cast<float>(minC[d] + delta * static_cast<NumericType>(dims[d]));
    }
    sourceArea_ = 1;
    for (int d = 0; d < D - 1; ++d)
      sourceArea_ *= delta * static_cast<NumericType>(dims[d]);

    primCell_.clear();
    primIdx_.clear();
    walkOut_.clear();
    for (size_t flat = 0; flat < sites; ++flat) {
      size_t rem = flat;
      std::array<int, D> idx{};
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lattice_->cellId(idx);
      if (id < 0 || !band[id] || (*fill_)[id] <= NumericType(0))
        continue;
      Vec3Df p{0.f, 0.f, 0.f};
      for (int d = 0; d < D; ++d)
        p[d] = static_cast<float>(minC[d] +
                                  delta * static_cast<NumericType>(idx[d]));
      grid.minPoints.push_back(p);
      grid.fills.push_back(static_cast<float>((*fill_)[id]));
      Vec3D<NumericType> up{0, 0, 0};
      up[D - 1] = 1; // where the gradient is degenerate the surface faces up
      // the estimator the caller asked for, not always Youngs: the Face and
      // narrow-gradient options exist to be compared against it, and forcing
      // one here makes that switch silently do nothing on the GPU
      const bool wide =
          estimator != viennacs::NormalEstimator::FillGradient;
      const auto n = interaction.gradientNormal(idx, up, wide);
      Vec3Df nf{0.f, 0.f, 0.f};
      for (int d = 0; d < D; ++d)
        nf[d] = static_cast<float>(n[d]);
      grid.normals.push_back(nf);

      // THE SAFE WALK-OUT for a ray re-emitted here, computed the way the CPU
      // does it: step along the dominant normal axis while cells still hold
      // material, and displace just past the last one. A fixed 2*delta cannot
      // know how thick the interface is: in a feature narrower than four
      // cells it lands the ray inside the OPPOSITE wall, which then culls it
      // and lets the ray escape through solid material.
      int axis = 0;
      NumericType steepest = 0;
      for (int d = 0; d < D; ++d)
        if (std::abs(n[d]) > steepest) {
          steepest = std::abs(n[d]);
          axis = d;
        }
      const int outward = n[axis] > NumericType(0) ? 1 : -1;
      int clear = 1;
      auto probe = idx;
      for (int stepOut = 0; stepOut < 8; ++stepOut) {
        probe[axis] += outward;
        const int nid = lattice_->cellId(probe);
        if (nid < 0 || (*fill_)[nid] <= NumericType(1e-9))
          break;
        ++clear;
      }
      walkOut_.push_back(static_cast<float>(
          delta * (static_cast<NumericType>(clear) + NumericType(1e-3))));

      primCell_.push_back(id);
      primIdx_.push_back(idx);
    }

    tracer_.setGeometry(grid);
    // the restart displaces the origin past the interface (in the reflection
    // callable), so the arming distance is only a self-intersection guard
    // the neutral's reflection callable displaces the origin past the
    // interface, so this is only a self-intersection guard
    tracer_.setArmingDistance(1e-3f * grid.gridDelta);
    std::vector<float> materialIds(primCell_.size(), 1.f);
    tracer_.setMaterialIds(materialIds);

    if (ionConfigured_ && effectiveMaterial != nullptr) {
      ionTracer_.setGeometry(grid);
      // the ion re-emits SPECULARLY off the wall it grazed, not back into
      // the interface, so the guard alone suffices; and its near-unity
      // sticking ends most rays at the first hit anyway
      // The ion's reflection callable does NOT displace its origin (the
      // neutral's does), so this gate is its only protection against
      // re-interacting inside the interface it just left. Without it an ion
      // burns its energy on immediate self-hits and never reaches a trench
      // floor: measured, the floor stops advancing entirely.
      ionTracer_.setArmingDistance(1.1f * grid.gridDelta);
      std::vector<float> legacyIds(primCell_.size());
      for (size_t p = 0; p < primCell_.size(); ++p) {
        const auto m =
            MaterialMap::mapToMaterial((*effectiveMaterial)[primCell_[p]]);
        legacyIds[p] = static_cast<float>(Material(m).legacyId());
      }
      ionTracer_.setMaterialIds(legacyIds);
    }

    ops_.prepareTransport(); // the host-side area cache for the operators
  }

  /// One neutral species: per-cell sticking in, incident flux density per
  /// cell out, in the CPU convention.
  std::vector<NumericType> trace(size_t numRays, NumericType sourceFlux,
                                 const std::vector<NumericType> &sticking,
                                 unsigned seed,
                                 int smoothingNeighbors = 1) {
    // two per-cell arrays, laid out [sticking | walkOut] as the device
    // indexes them: slot 0 at primID, slot 1 at numElements + primID
    std::vector<float> cellData(2 * primCell_.size());
    for (size_t p = 0; p < primCell_.size(); ++p) {
      cellData[p] = static_cast<float>(sticking[primCell_[p]]);
      cellData[primCell_.size() + p] = walkOut_[p];
    }
    stickingBuffer_.allocUpload(cellData);
    tracer_.setElementData(stickingBuffer_, 2);

    tracer_.setNumberOfRaysFixed(numRays);
    tracer_.setRngSeed(seed);
    tracer_.apply();
    const auto raw = tracer_.getFlux(0, 0);

    const NumericType rayRate =
        sourceFlux * sourceArea_ / static_cast<NumericType>(numRays);
    std::vector<NumericType> spread(fill_->size(), NumericType(0));
    for (size_t p = 0; p < raw.size(); ++p)
      if (raw[p] > 0)
        ops_.deposit(spread, primIdx_[p],
                     static_cast<NumericType>(raw[p]) * rayRate);

    const NumericType delta = lattice_->gridDelta();
    NumericType faceArea = 1;
    for (int d = 0; d < D - 1; ++d)
      faceArea *= delta;
    std::vector<NumericType> flux(fill_->size(), NumericType(0));
    const auto &dims = lattice_->dims();
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);
    for (size_t flat = 0; flat < sites; ++flat) {
      size_t rem = flat;
      std::array<int, D> idx{};
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lattice_->cellId(idx);
      if (id < 0 || spread[id] <= NumericType(0))
        continue;
      const NumericType area = ops_.areaAt(idx);
      if (area > NumericType(1e-2) * faceArea)
        flux[id] = spread[id] / area;
    }
    ops_.smooth(flux, smoothingNeighbors);
    return flux;
  }

  /// The ion: yield-weighted flux per channel, per cell, in the CPU ion's
  /// convention -- spread, area-normalised with the same sliver guard,
  /// smoothed one ring.
  std::vector<std::vector<NumericType>>
  traceIon(size_t numRays, unsigned seed) {
    ionTracer_.setNumberOfRaysFixed(numRays);
    ionTracer_.setRngSeed(seed);
    ionTracer_.apply();

    // the geometric share only; each channel is scaled by its own species
    const NumericType rayShare =
        sourceArea_ / static_cast<NumericType>(numRays);
    const NumericType delta = lattice_->gridDelta();
    NumericType faceArea = 1;
    for (int d = 0; d < D - 1; ++d)
      faceArea *= delta;
    const auto &dims = lattice_->dims();
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);

    std::vector<std::vector<NumericType>> channels(
        nChannels_, std::vector<NumericType>(fill_->size(), NumericType(0)));
    for (size_t c = 0; c < nChannels_; ++c) {
      const NumericType rayRate = rayShare * channelFlux_[c];
      const auto raw = ionTracer_.getFlux(0, static_cast<int>(c));
      std::vector<NumericType> spread(fill_->size(), NumericType(0));
      for (size_t p = 0; p < raw.size(); ++p)
        if (raw[p] > 0)
          ops_.deposit(spread, primIdx_[p],
                       static_cast<NumericType>(raw[p]) * rayRate);
      for (size_t flat = 0; flat < sites; ++flat) {
        size_t rem = flat;
        std::array<int, D> idx{};
        for (int d = 0; d < D; ++d) {
          idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
          rem /= static_cast<size_t>(dims[d]);
        }
        const int id = lattice_->cellId(idx);
        if (id < 0 || spread[id] <= NumericType(0))
          continue;
        const NumericType area = ops_.areaAt(idx);
        if (area > NumericType(1e-2) * faceArea)
          channels[c][id] = spread[id] / area;
      }
      ops_.smooth(channels[c], 1);
    }
    return channels;
  }
};

} // namespace viennaps

#endif // VIENNACORE_COMPILE_GPU
