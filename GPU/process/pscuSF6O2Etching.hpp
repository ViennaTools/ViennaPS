#pragma once

#include <context.hpp>

#include <SF6O2Etching.hpp>
#include <pscuProcess.hpp>
#include <pscuProcessModel.hpp>
#include <pscuSurfaceModel.hpp>

#include <curtParticle.hpp>

#include <utLaunchKernel.hpp>

extern "C" char embedded_SF6O2_pipeline[];

template <typename NumericType>
class pscuSF6O2SurfaceModel : public pscuSurfaceModel<NumericType> {
  const std::string processModuleName = "SF6O2ProcessKernels.ptx";
  const std::string calcEtchRateKernel = "calculateEtchRate";
  const std::string updateCoverageKernel = "updateCoverages";
  pscuContext context;
  NumericType totalIonFlux = 12.;
  NumericType totalEtchantFlux = 1.8e3;
  NumericType totalOxygenFlux = 1.0e2;

public:
  using pscuSurfaceModel<NumericType>::d_coverages;
  using pscuSurfaceModel<NumericType>::ratesIndexMap;
  using pscuSurfaceModel<NumericType>::coveragesIndexMap;

  pscuSF6O2SurfaceModel(pscuContext passedContext, const NumericType ionFlux,
                        const NumericType etchantFlux,
                        const NumericType oxygenFlux)
      : context(passedContext), totalIonFlux(ionFlux),
        totalEtchantFlux(etchantFlux), totalOxygenFlux(oxygenFlux) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    const int numCoverages = 2;
    d_coverages.alloc(numCoverages * numGeometryPoints * sizeof(NumericType));
    coveragesIndexMap.insert(std::make_pair("eCoverage", 0));
    coveragesIndexMap.insert(std::make_pair("oCoverage", 1));
  }

  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(utCudaBuffer &d_rates,
                      const std::vector<std::array<NumericType, 3>> &points,
                      const std::vector<NumericType> &materialIds) override {
    unsigned long numPoints = materialIds.size();
    updateCoverages(d_rates, numPoints);

    std::vector<NumericType> etchRate(numPoints, 0.);
    utCudaBuffer etchRateBuffer;
    utCudaBuffer materialIdsBuffer;
    utCudaBuffer pointCoordsBuffer;

    etchRateBuffer.alloc(numPoints * sizeof(NumericType));
    materialIdsBuffer.alloc_and_upload(materialIds);
    pointCoordsBuffer.alloc_and_upload(points);

    assert(d_rates.sizeInBytes / sizeof(NumericType) == numPoints * 5);
    CUdeviceptr rates = d_rates.d_pointer();
    assert(d_coverages.sizeInBytes / sizeof(NumericType) == numPoints * 2);
    CUdeviceptr coverages = d_coverages.d_pointer();
    CUdeviceptr erate = etchRateBuffer.d_pointer();
    CUdeviceptr matIds = materialIdsBuffer.d_pointer();
    CUdeviceptr coords = pointCoordsBuffer.d_pointer();

    // launch kernel
    void *kernelArgs[] = {
        &rates,     &coverages,    &coords,           &matIds,         &erate,
        &numPoints, &totalIonFlux, &totalEtchantFlux, &totalOxygenFlux};

    utLaunchKernel::launch(processModuleName, calcEtchRateKernel, kernelArgs,
                           context);

    etchRateBuffer.download(etchRate.data(), numPoints);

    // clean up
    etchRateBuffer.free();
    materialIdsBuffer.free();

    return psSmartPointer<std::vector<NumericType>>::New(std::move(etchRate));
  }

  void updateCoverages(utCudaBuffer &d_rates,
                       unsigned long numPoints) override {
    assert(d_rates.sizeInBytes / sizeof(NumericType) == numPoints * 5);
    CUdeviceptr rates = d_rates.d_pointer();
    assert(d_coverages.sizeInBytes / sizeof(NumericType) == numPoints * 2);
    CUdeviceptr coverages = d_coverages.d_pointer();

    // launch kernel
    void *kernelArgs[] = {&rates,        &coverages,        &numPoints,
                          &totalIonFlux, &totalEtchantFlux, &totalOxygenFlux};

    utLaunchKernel::launch(processModuleName, updateCoverageKernel, kernelArgs,
                           context);
  }
};
