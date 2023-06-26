#pragma once

#include <context.hpp>

#include <pscuProcess.hpp>
#include <pscuProcessModel.hpp>
#include <pscuSurfaceModel.hpp>

#include <curtParticle.hpp>

#include <utLaunchKernel.hpp>

extern "C" char embedded_deposition_pipeline[];

template <typename NumericType>
class IonMillingSurfaceModel : public pscuSurfaceModel<NumericType> {
  pscuContext context;

public:
  IonMillingSurfaceModel(pscuContext passedContext) : context(passedContext) {}

  psSmartPointer<std::vector<NumericType>>
  calculateVelocities(utCudaBuffer &d_rates,
                      const std::vector<std::array<NumericType, 3>> &points,
                      const std::vector<NumericType> &materialIds) override {
    unsigned long numPoints = materialIds.size();
    std::vector<NumericType> etchRate(numPoints, 1.);
    d_rates.download(etchRate.data(), numPoints);
    return psSmartPointer<std::vector<NumericType>>::New(etchRate);
  }
};

class pscuIonMilling {
  psSmartPointer<psDomain<NumericType, DIM>> geometry;
  NumericType rate;
  NumericType processTime;
  NumericType RF;
  NumericType cosineExponent;
  pscuContext context;
  int printIntermediate;
  int periodicBoundary;
  int raysPerPoint;

public:
  pscuIonMilling(psSmartPointer<psDomain<NumericType, DIM>> passedGeometry,
                 const NumericType passedRate, const NumericType passedTime,
                 const NumericType passedRF, const NumericType passedExponent,
                 pscuContext passedContext, const int passedPrint = 0,
                 const int passedPeriodic = 0, const int passedRays = 5000)
      : geometry(passedGeometry), rate(passedRate), processTime(passedTime),
        RF(passedRF), cosineExponent(passedExponent), context(passedContext),
        printIntermediate(passedPrint), periodicBoundary(passedPeriodic),
        raysPerPoint(passedRays) {}

  void apply() {
    curtParticle<NumericType> depoParticle{
        .name = "ion", .ionRF = RF, .cosineExponent = cosineExponent};
    depoParticle.dataLabels.push_back("ionRate");

    auto surfModel =
        psSmartPointer<IonMillingSurfaceModel<NumericType>>::New(context);
    auto velField = psSmartPointer<psDefaultVelocityField<NumericType>>::New();
    auto model = psSmartPointer<pscuProcessModel<NumericType>>::New();

    model->insertNextParticleType(depoParticle);
    model->setSurfaceModel(surfModel);
    model->setVelocityField(velField);
    model->setProcessName("IonMilling");
    model->setPtxCode(embedded_deposition_pipeline);

    pscuProcess<NumericType, DIM> process(context);
    process.setDomain(geometry);
    process.setProcessModel(model);
    process.setNumberOfRaysPerPoint(raysPerPoint);
    process.setProcessDuration(processTime);
    process.apply();
  }
};
