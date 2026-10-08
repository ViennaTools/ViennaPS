#pragma once

#ifdef VIENNACORE_COMPILE_GPU

#include "psVoxelPMCParamsGPU.hpp"

#include <gpu/raygTraceCell.hpp>

#include <vcContext.hpp>
#include <vcCudaBuffer.hpp>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace viennaps {

using namespace viennacore;

/// The binary-cell PMC's particle transport on the GPU, through ViennaRay's
/// cell tracer, the same TraceCell the filling-fraction voxel chemistry uses.
/// A binary cell is simply a fill of 1. Per step the PMC hands over its
/// surface band (exposed solid cells, mask included) with each cell's state,
/// normals, restart distance and spreading candidates. The states stay on the
/// device for the whole step and the hit programs change them as reactions
/// happen, so F, O and the ions, launched in that order, each see what the
/// species before left. Each species is traced as an exact number of analog
/// rays, and every reaction also comes back as an event for the PMC to apply
/// on the host.
template <class NumericType, int D> class VoxelPMCGPU {
public:
  /// What the PMC hands over per step, one entry per band cell.
  struct Band {
    std::vector<std::array<int, D>> idx;
    std::vector<Vec3Df> minPoints; ///< box minimum corner
    std::vector<Vec3Df> normals;   ///< the estimator's unit normal
    std::vector<int> state;        ///< VoxelPMCStateGPU
    std::vector<int> step;         ///< cascade step, F bonded
    std::vector<float> fitNormal;  ///< 3 per cell, plane fit
    std::vector<float> walkOut;    ///< restart distance, < 0 cannot escape
    std::vector<unsigned char> normalValid; ///< the estimator gave a normal
    std::vector<int> hopStart;     ///< size + 1, CSR offsets
    std::vector<int> hopList;
    std::vector<unsigned char> hopCheb;
    void clear() {
      idx.clear(); minPoints.clear(); normals.clear(); state.clear();
      step.clear(); fitNormal.clear(); walkOut.clear(); normalValid.clear();
      hopStart.clear();
      hopList.clear(); hopCheb.clear();
    }
  };

private:
  std::shared_ptr<DeviceContext> context_;
  viennaray::gpu::TraceCell<float, D> neutral_;
  viennaray::gpu::TraceCell<float, D> ion_;
  VoxelPMCParamsGPU p_{};
  CudaBuffer pBuf_;
  CudaBuffer idxBuf_, stateBuf_, stepBuf_, fitBuf_, walkBuf_, validBuf_;
  CudaBuffer hopStartBuf_, hopListBuf_, hopChebBuf_;
  CudaBuffer evBuf_, ionEvBuf_, countBuf_, ctrBuf_;
  unsigned int capacity_;
  size_t bandSize_ = 0;

  static void setUp(viennaray::gpu::TraceCell<float, D> &tracer,
                    std::shared_ptr<DeviceContext> context,
                    const std::string &name, float power,
                    const std::string &prefix) {
    viennaray::gpu::Particle<float> particle;
    particle.name = name;
    particle.dataLabels = {"unused"};
    particle.cosineExponent = power;
    particle.sticking = 1.f;
    std::unordered_map<std::string, unsigned int> pMap = {{name, 0}};
    std::vector<viennaray::gpu::CallableConfig> cMap = {
        {0, viennaray::gpu::CallableSlot::COLLISION,
         "__direct_callable__" + prefix + "Collision"},
        {0, viennaray::gpu::CallableSlot::REFLECTION,
         "__direct_callable__" + prefix + "Reflection"},
        {0, viennaray::gpu::CallableSlot::INIT,
         "__direct_callable__" + prefix + "Init"},
    };
    tracer.setCallables("ViennaPSCallableWrapper", context->modulePath);
    tracer.setParticleCallableMap({pMap, cMap});
    tracer.insertNextParticle(particle);
    tracer.setUseRandomSeeds(false);
    // once: compiles the modules and builds the pipeline; apply() rebuilds
    // only the shader binding table
    tracer.prepareParticlePrograms();
  }

  void resetOutputs() {
    const unsigned int zero = 0;
    countBuf_.upload(&zero, 1);
    std::vector<unsigned long long> z(VoxelPMCParamsGPU::numCounters, 0ull);
    ctrBuf_.upload(z.data(), z.size());
  }

  void uploadParams() { pBuf_.upload(&p_, 1); }

  unsigned int download(viennaray::gpu::TraceCell<float, D> &tracer,
                        std::array<unsigned long long,
                                   VoxelPMCParamsGPU::numCounters> &counters) {
    tracer.syncStreams();
    unsigned int count = 0;
    countBuf_.download(&count, 1);
    std::array<unsigned long long, VoxelPMCParamsGPU::numCounters> c{};
    ctrBuf_.download(c.data(), c.size());
    for (int i = 0; i < VoxelPMCParamsGPU::numCounters; ++i)
      counters[i] += c[i];
    return std::min(count, capacity_);
  }

  /// The first `count` entries of a larger buffer. CudaBuffer::download
  /// copies a whole buffer only.
  template <class T>
  static void downloadFirst(CudaBuffer &buffer, T *to, size_t count) {
    CUDA_CHECK(buffer.context->ch.cuMemcpyDtoH_(
        (void *)to, buffer.dPointer(), count * sizeof(T)));
  }

public:
  VoxelPMCGPU(std::shared_ptr<DeviceContext> context, float neutralPower,
              float ionPower, unsigned int capacity = 1u << 20)
      : context_(context), neutral_(context), ion_(context),
        capacity_(capacity) {
    setUp(neutral_, context_, "PMCNeutral", neutralPower, "pmcNeutral");
    setUp(ion_, context_, "PMCIon", ionPower, "pmcIon");
    pBuf_.alloc(sizeof(VoxelPMCParamsGPU));
    evBuf_.alloc(sizeof(int) * capacity_);
    ionEvBuf_.alloc(sizeof(float) * 4 * capacity_);
    countBuf_.alloc(sizeof(unsigned int));
    ctrBuf_.alloc(sizeof(unsigned long long) * VoxelPMCParamsGPU::numCounters);
    p_.D = D;
  }

  ~VoxelPMCGPU() {
    for (auto *b : {&pBuf_, &idxBuf_, &stateBuf_, &stepBuf_, &fitBuf_,
                    &walkBuf_, &validBuf_, &hopStartBuf_, &hopListBuf_,
                    &hopChebBuf_,
                    &evBuf_, &ionEvBuf_, &countBuf_, &ctrBuf_})
      b->free();
  }

  VoxelPMCGPU(const VoxelPMCGPU &) = delete;
  VoxelPMCGPU &operator=(const VoxelPMCGPU &) = delete;

  /// The model's own numbers, set once: spreading, cascade, re-hit rule, ion.
  VoxelPMCParamsGPU &params() { return p_; }

  /// Side-wall mirrors allowed per flight segment, the CPU's kMaxSideBounce:
  /// a segment that needs more is lost there, and so here. The device
  /// programs restart the count at every surface hit.
  void setMaxSideMirrors(unsigned m) {
    neutral_.setMaxBoundaryHits(m);
    ion_.setMaxBoundaryHits(m);
  }

  /// The surface as it stands at the start of a step.
  void prepare(const Band &band, const std::array<int, D> &dims,
               const std::array<NumericType, D> &minCorner,
               NumericType delta) {
    bandSize_ = band.idx.size();
    for (int d = 0; d < D; ++d)
      p_.dims[d] = dims[d];
    p_.bandSize = static_cast<int>(bandSize_);
    if (bandSize_ == 0)
      return;

    viennaray::gpu::CellGrid grid;
    grid.gridDelta = static_cast<float>(delta);
    for (int d = 0; d < 3; ++d) {
      grid.minimumExtent[d] = 0.f;
      grid.maximumExtent[d] = 0.f;
    }
    for (int d = 0; d < D; ++d) {
      grid.minimumExtent[d] = static_cast<float>(minCorner[d]);
      grid.maximumExtent[d] = static_cast<float>(
          minCorner[d] + delta * static_cast<NumericType>(dims[d]));
    }
    grid.minPoints = band.minPoints;
    grid.fills.assign(bandSize_, 1.f);
    grid.normals = band.normals;
    // the reflection programs restart the ray one cell out themselves, so the
    // arming distance is only a self-intersection guard
    neutral_.setGeometry(grid);
    neutral_.setArmingDistance(1e-3f * grid.gridDelta);
    ion_.setGeometry(grid);
    ion_.setArmingDistance(1e-3f * grid.gridDelta);
    const std::vector<float> materialIds(bandSize_, 1.f);
    neutral_.setMaterialIds(materialIds);
    ion_.setMaterialIds(materialIds);

    std::vector<int> idx3(3 * bandSize_, 0);
    for (size_t b = 0; b < bandSize_; ++b)
      for (int d = 0; d < D; ++d)
        idx3[3 * b + d] = band.idx[b][d];
    idxBuf_.allocUpload(idx3);
    stateBuf_.allocUpload(band.state);
    stepBuf_.allocUpload(band.step);
    fitBuf_.allocUpload(band.fitNormal);
    walkBuf_.allocUpload(band.walkOut);
    validBuf_.allocUpload(band.normalValid);
    hopStartBuf_.allocUpload(band.hopStart);
    if (band.hopList.empty()) {
      hopListBuf_.allocUpload(std::vector<int>{0});
      hopChebBuf_.allocUpload(std::vector<unsigned char>{0});
    } else {
      hopListBuf_.allocUpload(band.hopList);
      hopChebBuf_.allocUpload(band.hopCheb);
    }

    p_.bandIdx = (const int *)idxBuf_.dPointer();
    p_.bandState = (int *)stateBuf_.dPointer();
    p_.bandStep = (int *)stepBuf_.dPointer();
    p_.fitNormal = (const float *)fitBuf_.dPointer();
    p_.walkOut = (const float *)walkBuf_.dPointer();
    p_.normalValid = (const unsigned char *)validBuf_.dPointer();
    p_.hopStart = (const int *)hopStartBuf_.dPointer();
    p_.hopList = (const int *)hopListBuf_.dPointer();
    p_.hopCheb = (const unsigned char *)hopChebBuf_.dPointer();
    p_.events = (int *)evBuf_.dPointer();
    p_.ionEvents = (float *)ionEvBuf_.dPointer();
    p_.eventCount = (unsigned int *)countBuf_.dPointer();
    p_.eventCapacity = capacity_;
    p_.counters = (unsigned long long *)ctrBuf_.dPointer();
  }

  /// One neutral species (0 F, 1 O), `n` analog rays: the band ids of the
  /// reaction sites, in the order the device recorded them. The cell states
  /// on the device already include these reactions.
  void traceNeutral(int species, size_t n, unsigned int seed,
                    std::vector<int> &events,
                    std::array<unsigned long long,
                               VoxelPMCParamsGPU::numCounters> &counters) {
    events.clear();
    if (n == 0 || bandSize_ == 0)
      return;
    p_.species = species;
    resetOutputs();
    uploadParams();
    neutral_.setParameters(pBuf_.dPointer());
    neutral_.setNumberOfRaysFixed(n);
    neutral_.setRngSeed(seed);
    neutral_.apply();
    const unsigned int k = download(neutral_, counters);
    events.resize(k);
    if (k)
      downloadFirst(evBuf_, events.data(), k);
  }

  /// The ions, `n` analog rays: (band id, energy, cos theta, reflected) per
  /// impact on a non-mask cell.
  void traceIon(size_t n, unsigned int seed, std::vector<float> &events,
                std::array<unsigned long long,
                           VoxelPMCParamsGPU::numCounters> &counters) {
    events.clear();
    if (n == 0 || bandSize_ == 0)
      return;
    resetOutputs();
    uploadParams();
    ion_.setParameters(pBuf_.dPointer());
    ion_.setNumberOfRaysFixed(n);
    ion_.setRngSeed(seed);
    ion_.apply();
    const unsigned int k = download(ion_, counters);
    events.resize(4 * static_cast<size_t>(k));
    if (k)
      downloadFirst(ionEvBuf_, events.data(), 4 * static_cast<size_t>(k));
  }
};

} // namespace viennaps

#endif // VIENNACORE_COMPILE_GPU
