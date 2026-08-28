#pragma once

#include "psSurfaceChemistry.hpp"

#include <csVoxelAdvance.hpp>
#include "psVoxelIon.hpp"

#include <csVoxelFlux.hpp>

#ifdef VIENNACORE_COMPILE_GPU
#include "psVoxelFluxGPU.hpp"
#endif

#include <chrono>
#include <memory>

namespace viennaps {

using namespace viennacore;

/// Runs a chemical mechanism on a voxel grid instead of a level set.
///
/// The mechanism is untouched -- the same `ChemicalMechanism`, read from the
/// same file, solving the same coverages and returning the same velocity as
/// the level-set path. That is the point: with the chemistry provably
/// identical, a difference between the two arms can only come from how the
/// surface is represented, how flux reaches it, and how it moves.
///
/// What differs, and only this:
///
///   level set   ViennaRay over a disk or triangle surface; velocities
///               extended into a field and advected.
///   voxel       a grid walk over filling fractions; velocities applied as
///               volume, cell by cell.
///
/// The coverage solve runs PER CELL, because flux varies over the surface and
/// the steady state is not linear in it. Cells with no interface are skipped:
/// they have no surface to react on.
template <class NumericType, int D> class VoxelChemistry {
  ChemicalMechanism<NumericType> &mech_;
  const viennacs::LatticeMap<NumericType, D> &lattice_;
  std::vector<NumericType> &fill_;
  std::vector<int> material_; ///< per cell, as the cell set reported it

  viennacs::VoxelAdvance<NumericType, D> advance_;
  std::vector<int> effectiveMaterial_; ///< see resolveMaterials()

  size_t raysPerStep_ = 100000;
  viennacs::NormalEstimator estimator_ =
      viennacs::NormalEstimator::FillGradientYoungs;
  viennacs::TraversalEngine engine_ = viennacs::TraversalEngine::GridDDA;
  bool useGPU_ = false;
#ifdef VIENNACORE_COMPILE_GPU
  // lazily created on the first GPU trace; a cache, not part of the value
  mutable std::unique_ptr<VoxelFluxGPU<NumericType, D>> gpuFlux_;
#endif
  NumericType minimumArea_ = 0; // set from the grid spacing in the constructor
  int coverageIterations_ = 500;
  NumericType coverageTolerance_ = 1e-13;

public:
  VoxelChemistry(ChemicalMechanism<NumericType> &mech,
                 const viennacs::LatticeMap<NumericType, D> &lattice,
                 std::vector<NumericType> &fill,
                 std::vector<int> material)
      : mech_(mech), lattice_(lattice), fill_(fill),
        material_(std::move(material)), advance_(lattice) {
    resolveMaterials();
    NumericType faceArea = 1;
    for (int d = 0; d < D - 1; ++d)
      faceArea *= lattice.gridDelta();
    minimumArea_ = NumericType(1e-2) * faceArea;
  }

  void setRaysPerStep(size_t n) { raysPerStep_ = n; }

  /// The material labels as they NOW are. They evolve: refreshMaterials
  /// promotes gas cells the film has grown into, so a writer that copies the
  /// construction-time labels shows a Material field frozen at t = 0 while
  /// the fractions move -- which is how this getter came to exist.
  const std::vector<int> &materials() const { return material_; }
  void setNormalEstimator(viennacs::NormalEstimator e) { estimator_ = e; }
  void setTraversalEngine(viennacs::TraversalEngine e) { engine_ = e; }

  /// Trace the NEUTRAL transport on the GPU: the band as OptiX cell
  /// primitives, per-cell sticking uploaded beside it. The ion channel stays
  /// on the CPU for now; a step mixes the two freely, since each trace is
  /// independent. Without GPU support compiled in this is refused loudly
  /// rather than silently ignored.
  void setUseGPU(bool use) {
#ifdef VIENNACORE_COMPILE_GPU
    useGPU_ = use;
#else
    if (use)
      Logger::getInstance()
          .addWarning("VoxelChemistry: built without GPU support; the flux "
                      "stays on the CPU.")
          .print();
    useGPU_ = false;
#endif
  }
  void setCoverageParameters(int iterations, NumericType tolerance) {
    coverageIterations_ = iterations;
    coverageTolerance_ = tolerance;
  }

  /// Keeps the material labels abreast of a moving surface, then re-resolves
  /// the interface. Called at the start of every step.
  ///
  /// The labels come from the cell set at construction, where everything above
  /// the initial surface is GAS. A deposit grows INTO that region, and a label
  /// resolved once at construction reaches only a few rings up: the SiGe/Si
  /// film grew exactly three cells -- the ring-search depth -- and froze, its
  /// selective mechanism reading `default: 0` off the GAS label above. So a
  /// gas-labelled cell that has acquired material takes the label of the
  /// solid it grew from, one frontier ring per step, which is as fast as a
  /// surface can move.
  ///
  /// A deposit is thereby labelled as its SUBSTRATE, which is right when film
  /// and substrate are the same material and an approximation when they are
  /// not; labelling by the mechanism's own depositing solid is the proper
  /// extension for multi-solid chemistries.
  void refreshMaterials() {
    const auto gas = static_cast<int>(Material::GAS);
    const auto &dims = lattice_.dims();
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);

    const std::vector<int> before = material_; // promote off a snapshot
#pragma omp parallel for schedule(dynamic, 256)
    for (long long flat = 0; flat < static_cast<long long>(sites); ++flat) {
      std::array<int, D> idx{};
      size_t rem = static_cast<size_t>(flat);
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lattice_.cellId(idx);
      if (id < 0)
        continue;
      // THE RULE: filling fraction >= 0.5 is material, < 0.5 is gas. Label
      // and geometry are then one statement -- the Material field equals the
      // solid at the 0.5 iso, and a residue cell at fill 0.03 can never sit
      // in the void labelled silicon. Chemistry loses nothing: a hit in a
      // sub-0.5 surface cell takes its constants through the
      // effectiveMaterial resolution below, exactly as gas-labelled
      // interface cells always have.
      if (before[id] != gas && fill_[id] < NumericType(0.5)) {
        material_[id] = gas;
        continue;
      }
      if (before[id] != gas || fill_[id] < NumericType(0.5))
        continue;
      for (int ring = 1; ring <= 2; ++ring) {
        const int width = 2 * ring + 1;
        int span = 1;
        for (int d = 0; d < D; ++d)
          span *= width;
        bool done = false;
        for (int q = 0; q < span && !done; ++q) {
          std::array<int, D> probe = idx;
          int r = q;
          for (int d = 0; d < D; ++d) {
            probe[d] += r % width - ring;
            r /= width;
          }
          const int nid = lattice_.cellId(probe);
          if (nid >= 0 && before[nid] != gas) {
            material_[id] = before[nid];
            done = true;
          }
        }
        if (done)
          break;
      }
    }
    resolveMaterials();
  }

  /// The material a surface cell should react as.
  ///
  /// A cell set built to span the gas region labels everything above the
  /// surface GAS, and the interface straddles that boundary -- so roughly half
  /// the cells carrying interface area are labelled GAS. Reacting them as GAS
  /// is wrong: a surface cell is the surface OF something, and that something
  /// is the solid beneath it. It matters most for a selective mechanism, whose
  /// `materials:` block names the substrate and leaves `default: 0`. On the
  /// SiGe/Si stack this was the difference between 20 nm of growth and 1 nm:
  /// the interface cells hit the default, stuck nothing, and reacted at zero
  /// rate. The level-set arm never sees this, because its surface points carry
  /// the material of the material below them.
  ///
  /// So a GAS cell holding interface takes the material of the nearest cell
  /// that is not gas, searched outward by rings.
  void resolveMaterials() {
    const auto gas = static_cast<int>(Material::GAS);
    effectiveMaterial_ = material_;
    const auto &dims = lattice_.dims();

    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);

    // Each cell's coverage solve is independent of every other, so this runs
    // in parallel like the transport above it.
#pragma omp parallel for schedule(dynamic, 64)
    for (long long flat = 0; flat < static_cast<long long>(sites); ++flat) {
      std::array<int, D> idx{};
      size_t rem = static_cast<size_t>(flat);
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lattice_.cellId(idx);
      if (id < 0 || material_[id] != gas)
        continue;

      int found = -1;
      for (int ring = 1; ring <= 3 && found < 0; ++ring) {
        const int width = 2 * ring + 1;
        int span = 1;
        for (int d = 0; d < D; ++d)
          span *= width;
        for (int s = 0; s < span && found < 0; ++s) {
          std::array<int, D> probe = idx;
          int r = s;
          for (int d = 0; d < D; ++d) {
            probe[d] += r % width - ring;
            r /= width;
          }
          const int nid = lattice_.cellId(probe);
          if (nid >= 0 && material_[nid] != gas)
            found = material_[nid];
        }
      }
      if (found >= 0)
        effectiveMaterial_[id] = found;
    }
  }

  struct StepReport {
    size_t surfaceCells = 0;
    NumericType meanVelocity = 0;
    NumericType minVelocity = 0;
    NumericType maxVelocity = 0;
    NumericType volumeMoved = 0;
    NumericType volumeLost = 0;
    // Where the step's wall time went. The benchmark against the level set
    // is only interpretable with this split: transport is what the traversal
    // engines compete on, while advance and relabel sweep the whole lattice
    // and dilute any transport win until they are band-limited.
    double secondsTransport = 0; ///< neutral and ion Monte Carlo, together
    double secondsChemistry = 0; ///< per-cell coverage solve and rate laws
    double secondsAdvance = 0;   ///< moving the filling fractions
    double secondsRelabel = 0;   ///< refreshing the material labels
  };

  /// The incident flux of every traced gas species, per cell.
  ///
  /// INCIDENT, not absorbed. The level-set particle deposits its full ray
  /// weight and lets the rate law apply the sticking, so this must do the
  /// same: depositing `weight * sticking` here would apply the sticking twice,
  /// and for silane, whose sticking is 4e-4 at 900 K, that is a rate three
  /// orders of magnitude too small. The sticking still governs how much weight
  /// survives to be re-emitted -- it is the deposit that must be unweighted.
  ///
  /// Untraced species are left empty here; `step` fills them from
  /// `sourceFluxes`, which knows to weight an ion channel by its yield.
  std::vector<std::vector<NumericType>>
  traceFluxes(unsigned seed,
              const std::vector<std::vector<NumericType>> &theta) const {
    const size_t nGas = mech_.gas.size();
    std::vector<std::vector<NumericType>> gamma(
        nGas, std::vector<NumericType>(fill_.size(), NumericType(0)));

#ifdef VIENNACORE_COMPILE_GPU
    if (useGPU_) {
      if (!gpuFlux_) {
        gpuFlux_ = std::make_unique<VoxelFluxGPU<NumericType, D>>(
            lattice_, fill_, DeviceContext::createContext());
        gpuFlux_->configureIon(mech_); // a no-op for an ionless mechanism
      }
      // once per step: the fills do not move between the species traces
      gpuFlux_->prepareGeometry(estimator_, &effectiveMaterial_);
    }
#endif

    for (size_t g = 0; g < nGas; ++g) {
      const auto &species = mech_.gas[g];
      if (!species.traced || species.isIonChannel)
        continue; // ion channels are traced below, by the ion itself
      // Sticking per cell -- and it is NOT s0. The level-set particle re-emits
      // with s0 * theta_free^n, evaluated from the coverages at the point it
      // hit (psSurfaceChemistry.hpp, ParticleNeutral::surfaceReflection). A
      // saturated surface therefore reflects far more than s0 suggests, and
      // that is what carries flux to the floor of a feature.
      //
      // Passing s0 alone made this arm stick rays that the level-set arm would
      // have reflected, so it saw line of sight where the level set saw a
      // near-conformal supply. On silane at s0 = 1 the surface saturates to
      // theta_free ~ 0.17, so theta_free^2 ~ 0.03: effectively non-sticking,
      // and the difference between a step coverage of 0.98 and one of 0.50.
      const int freeExp = mech_.gas[g].stickingFreeSiteExponent;
      const int site = mech_.gas[g].stickingSite;
      std::vector<NumericType> sticking(fill_.size(), NumericType(0));
      for (size_t c = 0; c < fill_.size(); ++c) {
        NumericType s = mech_.stickingOf(
            static_cast<int>(g),
            MaterialMap::mapToMaterial(effectiveMaterial_[c]));
        if (freeExp > 0 && !theta[c].empty()) {
          const auto free = mech_.freeFractions(theta[c]);
          const NumericType f =
              site < static_cast<int>(free.size()) ? free[site] : NumericType(1);
          for (int k = 0; k < freeExp; ++k)
            s *= f;
        }
        sticking[c] = s;
      }

#ifdef VIENNACORE_COMPILE_GPU
      if (useGPU_) {
        gamma[g] = gpuFlux_->trace(raysPerStep_, species.sourceFlux, sticking,
                                   seed + static_cast<unsigned>(g));
        continue;
      }
#endif
      viennacs::VoxelFlux<NumericType, D> flux(lattice_, fill_, estimator_);
      flux.setTraversalEngine(engine_);
      const auto result =
          flux.trace(raysPerStep_, species.sourceFlux, sticking,
                     NumericType(1), seed + static_cast<unsigned>(g));
      gamma[g] = result.flux;
    }

    // The ion. Its channels carry a yield that depends on the energy it still
    // has and on the angle it strikes at, so they cannot be taken from
    // sourceFluxes once there is a feature: that value is the yield at normal
    // incidence, which is exact on a blanket and unshadowed everywhere else.
    if (!mech_.ionYields.empty()) {
#ifdef VIENNACORE_COMPILE_GPU
      if (useGPU_ && gpuFlux_ && gpuFlux_->ionConfigured()) {
        const auto channels = gpuFlux_->traceIon(
            raysPerStep_,
            mech_.gas[mech_.ionYields[0].gasIndex].sourceFlux, seed + 977u);
        for (size_t c = 0; c < channels.size(); ++c)
          gamma[mech_.ionYields[c].gasIndex] = channels[c];
        return gamma;
      }
#endif
      VoxelIonFlux<NumericType, D> ion(mech_, lattice_, fill_,
                                       effectiveMaterial_, estimator_);
      ion.setTraversalEngine(engine_);
      const auto channels = ion.trace(raysPerStep_, seed + 977u);
      for (size_t c = 0; c < channels.size(); ++c)
        gamma[mech_.ionYields[c].gasIndex] = channels[c];
    }
    return gamma;
  }

  /// One step: trace, solve the chemistry on every surface cell, move the
  /// surface.
  StepReport step(NumericType dt, std::vector<std::vector<NumericType>> &theta,
                  unsigned seed = 1) {
    const auto tTransport0 = std::chrono::steady_clock::now();
    const auto gamma = traceFluxes(seed, theta);
    const auto tTransport1 = std::chrono::steady_clock::now();
    const size_t nCov = mech_.coverageNames.size();
    const size_t nGas = mech_.gas.size();

    std::vector<NumericType> velocity(fill_.size(), NumericType(0));
    // the material each cell's velocity is computed FOR -- the advance may
    // only place a share into that material
    std::vector<int> velMat(fill_.size(), static_cast<int>(Material::GAS));

    StepReport report;
    report.minVelocity = std::numeric_limits<NumericType>::max();
    report.maxVelocity = std::numeric_limits<NumericType>::lowest();

    const auto &dims = lattice_.dims();
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);

    // Only the interface band can hold surface cells: outside it the area is
    // provably zero, and the loop below would reject every cell anyway --
    // after paying the stencil for each. One linear pass buys the skip.
    const auto band = advance_.interfaceBand(fill_);

    // Each cell's coverage solve is independent of every other, so this runs
    // in parallel like the transport above it.
#pragma omp parallel for schedule(dynamic, 64)
    for (long long flat = 0; flat < static_cast<long long>(sites); ++flat) {
      std::array<int, D> idx{};
      size_t rem = static_cast<size_t>(flat);
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lattice_.cellId(idx);
      if (id < 0)
        continue;
      if (!band[id])
        continue; // provably zero interface area out here
      // A cell has to hold enough interface to be a surface. Flux density is
      // a rate over an area, so a cell with a sliver of area divides a small
      // number by a smaller one and reports a flux far above anything
      // incident -- which the chemistry then believes. On silane that put a
      // cell at 2.5 nm/s against a blanket rate of 0.07.
      const NumericType area = advance_.interfaceArea(fill_, idx);
      if (area <= minimumArea_)
        continue;

      const Material material =
          MaterialMap::mapToMaterial(effectiveMaterial_[id]);
      velMat[id] = effectiveMaterial_[id];
      // Start from the analytic incident fluxes, which carry the ion yield at
      // normal incidence for any yield channel, then replace the entries a
      // trace actually resolved.
      auto cellGamma = mech_.sourceFluxes(material);
      for (size_t g = 0; g < nGas; ++g)
        if (mech_.gas[g].traced && !gamma[g].empty())
          cellGamma[g] = gamma[g][id];

      const auto k = mech_.rateConstantsFor(material);
      if (nCov > 0) {
        // Continue from this cell's previous coverages: they are close to the
        // new steady state, so Newton converges in a few steps instead of
        // climbing from a bare surface every time.
        mech_.solveCoverages(cellGamma, k, theta[id], coverageIterations_,
                             coverageTolerance_);
      }
      velocity[id] = mech_.growthRate(cellGamma, k, theta[id], material);

#pragma omp critical
      {
        ++report.surfaceCells;
        report.meanVelocity += velocity[id];
        report.minVelocity = std::min(report.minVelocity, velocity[id]);
        report.maxVelocity = std::max(report.maxVelocity, velocity[id]);
      }
    }
    if (report.surfaceCells)
      report.meanVelocity /= static_cast<NumericType>(report.surfaceCells);
    else
      report.minVelocity = report.maxVelocity = 0;

    const auto tChemistry = std::chrono::steady_clock::now();
    const auto moved = advance_.apply(fill_, velocity, dt, &material_,
                                      static_cast<int>(Material::GAS), &velMat);
    const auto tAdvance = std::chrono::steady_clock::now();
    // AFTER the advance, so the labels always describe the fill that exists:
    // refreshed before the trace instead, the final step's advance leaves
    // emptied cells still labelled solid in anything written afterwards --
    // which is how a Material view came to show un-etched matter over
    // fractions that had left. The first step's trace is covered by the
    // resolution the constructor runs.
    refreshMaterials();
    const auto tRelabel = std::chrono::steady_clock::now();
    using Sec = std::chrono::duration<double>;
    report.secondsTransport = Sec(tTransport1 - tTransport0).count();
    report.secondsChemistry = Sec(tChemistry - tTransport1).count();
    report.secondsAdvance = Sec(tAdvance - tChemistry).count();
    report.secondsRelabel = Sec(tRelabel - tAdvance).count();
    report.volumeMoved = moved.volumeRequested;
    report.volumeLost = moved.volumeLost;
    return report;
  }

  /// Coverages for every cell, starting from a bare surface.
  std::vector<std::vector<NumericType>> makeCoverages() const {
    return std::vector<std::vector<NumericType>>(
        fill_.size(),
        std::vector<NumericType>(mech_.coverageNames.size(), NumericType(0)));
  }
};

} // namespace viennaps
