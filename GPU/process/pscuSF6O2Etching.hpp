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
class SurfaceModel : public pscuSurfaceModel<NumericType> {
  const std::string processModuleName = "SF6O2ProcessKernels.ptx";
  const std::string calcEtchRateKernel = "calculateEtchRate";
  const std::string updateCoverageKernel = "updateCoverages";
  pscuContext context;

public:
  using pscuSurfaceModel<NumericType>::d_processParams;
  using pscuSurfaceModel<NumericType>::d_coverages;
  using pscuSurfaceModel<NumericType>::ratesIndexMap;
  using pscuSurfaceModel<NumericType>::coveragesIndexMap;

  SurfaceModel(pscuContext passedContext) : context(passedContext) {}

  void initializeCoverages(unsigned numGeometryPoints) override {
    const int numCoverages = 2;
    d_coverages.alloc(numCoverages * numGeometryPoints * sizeof(NumericType));
    coveragesIndexMap.insert(std::make_pair("eCoverage", 0));
    coveragesIndexMap.insert(std::make_pair("oCoverage", 1));
  }

  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(utCudaBuffer &d_rates,
                      const std::vector<NumericType> &materialIds) override {
    unsigned long numPoints = materialIds.size();
    updateCoverages(d_rates, numPoints);

    std::vector<NumericType> etchRate(numPoints, 1.);
    utCudaBuffer etchRateBuffer;
    utCudaBuffer materialIdsBuffer;

    etchRateBuffer.alloc(numPoints * sizeof(NumericType));
    materialIdsBuffer.alloc_and_upload(materialIds);

    assert(d_rates.sizeInBytes / sizeof(NumericType) == numPoints * 5);
    CUdeviceptr rates = d_rates.d_pointer();
    assert(d_coverages.sizeInBytes / sizeof(NumericType) == numPoints * 2);
    CUdeviceptr coverages = d_coverages.d_pointer();
    CUdeviceptr erate = etchRateBuffer.d_pointer();
    CUdeviceptr matIds = materialIdsBuffer.d_pointer();

    // launch kernel
    void *kernel_args[] = {&rates, &coverages, &matIds, &erate, &numPoints};

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
    assert(d_rates.sizeInBytes / sizeof(NumericType) == numPoints * 5);
    CUdeviceptr rates = d_rates.d_pointer();
    assert(d_coverages.sizeInBytes / sizeof(NumericType) == numPoints * 2);
    CUdeviceptr coverages = d_coverages.d_pointer();

    // launch kernel
    void *kernel_args[] = {&rates, &coverages, &numPoints};

    utLaunchKernel::launch(processModuleName, updateCoverageKernel, kernel_args,
                           context);
  }
};

class pscuSF6O2Etching {
  psSmartPointer<psDomain<NumericType, D>> geometry;
  NumericType processTime;
  pscuContext context;
  int printIntermediate;
  int periodicBoundary;
  int raysPerPoint;

public:
  pscuSF6O2Etching(psSmartPointer<psDomain<NumericType, D>> passedGeometry,
                   const NumericType passedTime, pscuContext passedContext,
                   const int passedPrint = 0, const int passedPeriodic = 0,
                   const int passedRays = 5000)
      : geometry(passedGeometry), processTime(passedTime),
        context(passedContext), printIntermediate(passedPrint),
        periodicBoundary(passedPeriodic), raysPerPoint(passedRays) {}

  void apply() {
    curtParticle<NumericType> ion{"ion", 3};
    ion.dataLabels.push_back("ionSputteringRate");
    ion.dataLabels.push_back("ionEnhancedRate");
    ion.dataLabels.push_back("oxygenSputteringRate");

    curtParticle<NumericType> etchant{"etchant", 1};
    etchant.dataLabels.push_back("etchantRate");

    curtParticle<NumericType> oxygen{"oxygen", 1};
    oxygen.dataLabels.push_back("oxygenRate");

    auto surfModel = psSmartPointer<SurfaceModel<NumericType>>::New(context);
    auto velField = psSmartPointer<SF6O2VelocityField<NumericType>>::New(0);
    auto model = psSmartPointer<pscuProcessModel<NumericType>>::New();

    model->insertNextParticleType(ion);
    model->insertNextParticleType(etchant);
    model->insertNextParticleType(oxygen);
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);
    model->setProcessName("SF6O2Etching");
    model->setPtxCode(embedded_SF6O2_pipeline);

    pscuProcess<NumericType, D> process(context);
    process.setDomain(geometry);
    process.setProcessModel(model);
    process.setNumberOfRaysPerPoint(3000);
    process.setMaxCoverageInitIterations(10);
    process.setProcessDuration(processTime);
    process.apply();
  }
};
