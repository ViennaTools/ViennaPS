#pragma once

#include <context.hpp>

#include <SimpleDeposition.hpp>
#include <pscuProcess.hpp>
#include <pscuProcessModel.hpp>
#include <pscuSurfaceModel.hpp>

#include <curtParticle.hpp>

#include <utLaunchKernel.hpp>

extern "C" char embedded_deposition_pipeline[];

template <typename NumericType>
class DepoSurfaceModel : public pscuSurfaceModel<NumericType> {
  pscuContext context;

public:
  DepoSurfaceModel(pscuContext passedContext) : context(passedContext) {}

  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(utCudaBuffer &d_rates,
                      const std::vector<NumericType> &materialIds) override {
    unsigned long numPoints = materialIds.size();
    std::vector<NumericType> etchRate(numPoints, 1.);
    d_rates.download(etchRate.data(), numPoints);
    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }
};

class pscuDeposition {
  psSmartPointer<psDomain<NumericType, DIM>> geometry;
  NumericType rate;
  NumericType processTime;
  NumericType sticking;
  NumericType cosineExponent;
  pscuContext context;
  int printIntermediate;
  int periodicBoundary;
  int raysPerPoint;

public:
  pscuDeposition(psSmartPointer<psDomain<NumericType, DIM>> passedGeometry,
                 const NumericType passedRate, const NumericType passedTime,
                 const NumericType passedSticking,
                 const NumericType passedExponent, pscuContext passedContext,
                 const int passedPrint = 0, const int passedPeriodic = 0,
                 const int passedRays = 5000)
      : geometry(passedGeometry), rate(passedRate), processTime(passedTime),
        sticking(passedSticking), cosineExponent(passedExponent),
        context(passedContext), printIntermediate(passedPrint),
        periodicBoundary(passedPeriodic), raysPerPoint(passedRays) {}

  void apply() {
    curtParticle<NumericType> depoParticle{.name = "depoParticle",
                                           .sticking = sticking,
                                           .cosineExponent = cosineExponent};
    depoParticle.dataLabels.push_back("depoRate");

    auto surfModel =
        psSmartPointer<DepoSurfaceModel<NumericType>>::New(context);
    auto velField =
        psSmartPointer<SimpleDepositionVelocityField<NumericType>>::New();
    auto model = psSmartPointer<pscuProcessModel<NumericType>>::New();

    model->insertNextParticleType(depoParticle);
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);
    model->setProcessName("Deposition");
    model->setPtxCode(embedded_deposition_pipeline);

    pscuProcess<NumericType, DIM> process(context);
    process.setDomain(geometry);
    process.setProcessModel(model);
    process.setNumberOfRaysPerPoint(raysPerPoint);
    process.setProcessDuration(processTime * rate / sticking);
    process.apply();
  }
};
