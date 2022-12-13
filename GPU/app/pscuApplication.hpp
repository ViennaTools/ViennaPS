#pragma once

#include "context.hpp"

#include "Application.hpp"

#include "pscuDeposition.hpp"
#include "pscuSF6O2Etching.hpp"

class pscuApplication : public Application<3> {
  pscuContext context;

public:
  pscuApplication(int argc, char **argv) : Application(argc, argv) {
    std::cout << "Initializing CUDA and OptiX ... ";
    pscuCreateContext(context);
    std::cout << "success" << std::endl;
  }

protected:
  void runSimpleDeposition(
      psSmartPointer<psDomain<NumericType, D>> processGeometry,
      psSmartPointer<ApplicationParameters> processParams) override {
    pscuDeposition(processGeometry, processParams->rate,
                   processParams->processTime, processParams->sticking,
                   processParams->cosinePower, context,
                   processParams->printIntermediate,
                   processParams->periodicBoundary, processParams->raysPerPoint)
        .apply();
  }

  void runSF6O2Etching(
      psSmartPointer<psDomain<NumericType, D>> processGeometry,
      psSmartPointer<ApplicationParameters> processParams) override {
    pscuSF6O2Etching(processGeometry, processParams->processTime, context,
                     processParams->printIntermediate,
                     processParams->periodicBoundary,
                     processParams->raysPerPoint);
  }
};