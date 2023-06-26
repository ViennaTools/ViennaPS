#pragma once

#include <context.hpp>

#include <SF6O2Etching.hpp>
#include <pscuProcess.hpp>
#include <pscuProcessModel.hpp>
#include <pscuSurfaceModel.hpp>

#include <curtParticle.hpp>

#include <utLaunchKernel.hpp>

extern "C" char embedded_Fluorocarbon_pipeline[];

template <typename NumericType>
class pscuFluorocarbonSurfaceModel : public pscuSurfaceModel<NumericType> {
  const int numRates = 6;
  const int numCoverages = 3;

  const std::string processModuleName = "FluorocarbonProcessKernels.ptx";
  const std::string calcEtchRateKernel = "calculateEtchRate";
  const std::string updateCoverageKernel = "updateCoverages";
  pscuContext context;
  NumericType totalIonFlux;
  NumericType totalEtchantFlux;
  NumericType totalPolyFlux;
  NumericType FJ_ev;

public:
  using pscuSurfaceModel<NumericType>::d_processParams;
  using pscuSurfaceModel<NumericType>::d_coverages;
  using pscuSurfaceModel<NumericType>::ratesIndexMap;
  using pscuSurfaceModel<NumericType>::coveragesIndexMap;

  pscuFluorocarbonSurfaceModel(pscuContext passedContext,
                               const NumericType ionFlux,
                               const NumericType etchantFlux,
                               const NumericType polyFlux,
                               const NumericType passedTemperature)
      : context(passedContext), totalIonFlux(ionFlux),
        totalEtchantFlux(etchantFlux), totalPolyFlux(polyFlux),
        FJ_ev(2.7 * std::exp(-0.168 / (0.000086173324 * passedTemperature)) *
              etchantFlux) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    d_coverages.alloc(numCoverages * numGeometryPoints * sizeof(NumericType));
    coveragesIndexMap.insert(std::make_pair("eCoverage", 0));
    coveragesIndexMap.insert(std::make_pair("pCoverage", 1));
    coveragesIndexMap.insert(std::make_pair("peCoverage", 2));
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

    etchRateBuffer.alloc(numPoints * sizeof(NumericType));
    materialIdsBuffer.alloc_and_upload(materialIds);

    assert(d_rates.sizeInBytes / sizeof(NumericType) == numPoints * numRates);
    CUdeviceptr rates = d_rates.d_pointer();
    assert(d_coverages.sizeInBytes / sizeof(NumericType) ==
           numPoints * numCoverages);
    CUdeviceptr coverages = d_coverages.d_pointer();
    CUdeviceptr erate = etchRateBuffer.d_pointer();
    CUdeviceptr matIds = materialIdsBuffer.d_pointer();

    // launch kernel
    void *kernel_args[] = {
        &rates,        &coverages,        &matIds,        &erate, &numPoints,
        &totalIonFlux, &totalEtchantFlux, &totalPolyFlux, &FJ_ev};

    utLaunchKernel::launch(processModuleName, calcEtchRateKernel, kernel_args,
                           context);

    etchRateBuffer.download(etchRate.data(), numPoints);

    // clean up
    etchRateBuffer.free();
    materialIdsBuffer.free();

    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }

  void updateCoverages(utCudaBuffer &d_rates,
                       unsigned long numPoints) override {
    assert(d_rates.sizeInBytes / sizeof(NumericType) == numPoints * numRates);
    CUdeviceptr rates = d_rates.d_pointer();
    assert(d_coverages.sizeInBytes / sizeof(NumericType) ==
           numPoints * numCoverages);
    CUdeviceptr coverages = d_coverages.d_pointer();

    // launch kernel
    void *kernel_args[] = {&rates,        &coverages,        &numPoints,
                           &totalIonFlux, &totalEtchantFlux, &totalPolyFlux,
                           &FJ_ev};

    utLaunchKernel::launch(processModuleName, updateCoverageKernel, kernel_args,
                           context);
  }
};
