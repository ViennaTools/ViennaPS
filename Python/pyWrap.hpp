#pragma once

#define PYBIND11_DETAILED_ERROR_MESSAGES

// correct module name macro
#define TOKENPASTE_INTERNAL(x, y, z) x##y##z
#define TOKENPASTE(x, y, z) TOKENPASTE_INTERNAL(x, y, z)
#define STRINGIZE2(s) #s
#define STRINGIZE(s) STRINGIZE2(s)
#define VIENNAPS_MODULE_VERSION STRINGIZE(VIENNAPS_VERSION)

#include <optional>
#include <vector>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// all header files which define API functions
#include <psAdvectionCallback.hpp>
#include <psDomain.hpp>
#include <psExtrude.hpp>
#include <psGDSGeometry.hpp>
#include <psGDSReader.hpp>
#include <psPlanarize.hpp>
#include <psProcess.hpp>
#include <psProcessModel.hpp>
#include <psSurfaceModel.hpp>

// geometries
#include <psMakeFin.hpp>
#include <psMakeHole.hpp>
#include <psMakePlane.hpp>
#include <psMakeStack.hpp>
#include <psMakeTrench.hpp>

// visualization
#include <psToDiskMesh.hpp>
#include <psToSurfaceMesh.hpp>
#include <psWriteVisualizationMesh.hpp>

// models
#include <DirectionalEtching.hpp>
#include <FluorocarbonEtching.hpp>
#include <GeometricDistributionModels.hpp>
#include <IsotropicProcess.hpp>
#include <PlasmaDamage.hpp>
#include <SF6O2Etching.hpp>
#include <SimpleDeposition.hpp>
#include <StackRedeposition.hpp>
#include <TEOSDeposition.hpp>
#include <WetEtching.hpp>

// CellSet
#include <csDenseCellSet.hpp>

// Compact
#include <psKDTree.hpp>

// other
#include <lsDomain.hpp>
#include <rayParticle.hpp>
#include <rayReflection.hpp>
#include <rayTraceDirection.hpp>
#include <rayUtil.hpp>

// always use double for python export
typedef double T;
typedef std::vector<hrleCoordType> VectorHRLEcoord;
// get dimension from cmake define
constexpr int D = VIENNAPS_PYTHON_DIMENSION;
typedef psSmartPointer<psDomain<T, D>> DomainType;

PYBIND11_DECLARE_HOLDER_TYPE(Types, psSmartPointer<Types>)
PYBIND11_MAKE_OPAQUE(std::vector<T, std::allocator<T>>)

PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)

// define trampoline classes for interface functions
// ALSO NEED TO ADD TRAMPOLINE CLASSES FOR CLASSES
// WHICH HOLD REFERENCES TO INTERFACE(ABSTRACT) CLASSES

class PypsSurfaceModel : public psSurfaceModel<T> {
  using psSurfaceModel<T>::Coverages;
  using psSurfaceModel<T>::processParams;
  using psSurfaceModel<T>::getCoverages;
  using psSurfaceModel<T>::getProcessParameters;
  typedef std::vector<T> vect_type;

public:
  void initializeCoverages(unsigned numGeometryPoints) override {
    PYBIND11_OVERRIDE(void, psSurfaceModel<T>, initializeCoverages,
                      numGeometryPoints);
  }

  void initializeProcessParameters() override {
    PYBIND11_OVERRIDE(void, psSurfaceModel<T>, initializeProcessParameters, );
  }

  psSmartPointer<std::vector<T>>
  calculateVelocities(psSmartPointer<psPointData<T>> Rates,
                      const std::vector<std::array<T, 3>> &coordinates,
                      const std::vector<T> &materialIds) override {
    PYBIND11_OVERRIDE(psSmartPointer<std::vector<T>>, psSurfaceModel<T>,
                      calculateVelocities, Rates, coordinates, materialIds);
  }

  void updateCoverages(psSmartPointer<psPointData<T>> Rates,
                       const std::vector<T> &materialIds) override {
    PYBIND11_OVERRIDE(void, psSurfaceModel<T>, updateCoverages, Rates,
                      materialIds);
  }
};

// psAdvectionCallback
class PyAdvectionCallback : public psAdvectionCallback<T, D> {
protected:
  using ClassName = psAdvectionCallback<T, D>;

public:
  using ClassName::domain;

  bool applyPreAdvect(const T processTime) override {
    PYBIND11_OVERRIDE(bool, ClassName, applyPreAdvect, processTime);
  }

  bool applyPostAdvect(const T advectionTime) override {
    PYBIND11_OVERRIDE(bool, ClassName, applyPostAdvect, advectionTime);
  }
};

// Particle Class
template <int D> class psParticle : public rayParticle<psParticle<D>, T> {
  using ClassName = rayParticle<psParticle<D>, T>;

public:
  void surfaceCollision(T rayWeight, const rayTriple<T> &rayDir,
                        const rayTriple<T> &geomNormal,
                        const unsigned int primID, const int materialID,
                        rayTracingData<T> &localData,
                        const rayTracingData<T> *globalData,
                        rayRNG &Rng) override final {
    PYBIND11_OVERRIDE(void, ClassName, surfaceCollision, rayWeight, rayDir,
                      geomNormal, primID, materialID, localData, globalData,
                      Rng);
  }

  std::pair<T, rayTriple<T>>
  surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                    const rayTriple<T> &geomNormal, const unsigned int primID,
                    const int materialID, const rayTracingData<T> *globalData,
                    rayRNG &Rng) override final {
    using Pair = std::pair<T, rayTriple<T>>;
    PYBIND11_OVERRIDE(Pair, ClassName, surfaceReflection, rayWeight, rayDir,
                      geomNormal, primID, materialID, globalData, Rng);
  }

  void initNew(rayRNG &RNG) override final {
    PYBIND11_OVERRIDE(void, ClassName, initNew, RNG);
  }

  int getRequiredLocalDataSize() const override final {
    PYBIND11_OVERRIDE(int, ClassName, getRequiredLocalDataSize);
  }

  T getSourceDistributionPower() const override final {
    PYBIND11_OVERRIDE(T, ClassName, getSourceDistributionPower);
  }

  std::vector<std::string> getLocalDataLabels() const override final {
    PYBIND11_OVERRIDE(std::vector<std::string>, ClassName, getLocalDataLabels);
  }
};

// Default particle classes
template <int D>
class psDiffuseParticle : public rayParticle<psDiffuseParticle<D>, T> {
  using ClassName = rayParticle<psDiffuseParticle<D>, T>;

public:
  psDiffuseParticle(const T pStickingProbability, const T pCosineExponent,
                    const std::string &pDataLabel)
      : stickingProbability(pStickingProbability),
        cosineExponent(pCosineExponent), dataLabel(pDataLabel) {}

  void surfaceCollision(T rayWeight, const rayTriple<T> &rayDir,
                        const rayTriple<T> &geomNormal,
                        const unsigned int primID, const int materialID,
                        rayTracingData<T> &localData,
                        const rayTracingData<T> *globalData,
                        rayRNG &Rng) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }

  std::pair<T, rayTriple<T>>
  surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                    const rayTriple<T> &geomNormal, const unsigned int primID,
                    const int materialID, const rayTracingData<T> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionDiffuse<T, D>(geomNormal, Rng);
    return {stickingProbability, direction};
  }

  void initNew(rayRNG &RNG) override final {}

  int getRequiredLocalDataSize() const override final { return 1; }

  T getSourceDistributionPower() const override final { return cosineExponent; }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel};
  }

private:
  const T stickingProbability = 1.;
  const T cosineExponent = 1.;
  const std::string dataLabel = "flux";
};

class psSpecularParticle : public rayParticle<psSpecularParticle, T> {
  using ClassName = rayParticle<psSpecularParticle, T>;

public:
  psSpecularParticle(const T pStickingProbability, const T pCosineExponent,
                     const std::string &pDataLabel)
      : stickingProbability(pStickingProbability),
        cosineExponent(pCosineExponent), dataLabel(pDataLabel) {}

  void surfaceCollision(T rayWeight, const rayTriple<T> &rayDir,
                        const rayTriple<T> &geomNormal,
                        const unsigned int primID, const int materialID,
                        rayTracingData<T> &localData,
                        const rayTracingData<T> *globalData,
                        rayRNG &Rng) override final {
    localData.getVectorData(0)[primID] += rayWeight;
  }

  std::pair<T, rayTriple<T>>
  surfaceReflection(T rayWeight, const rayTriple<T> &rayDir,
                    const rayTriple<T> &geomNormal, const unsigned int primID,
                    const int materialID, const rayTracingData<T> *globalData,
                    rayRNG &Rng) override final {
    auto direction = rayReflectionSpecular<T>(rayDir, geomNormal);
    return {stickingProbability, direction};
  }

  void initNew(rayRNG &RNG) override final {}

  int getRequiredLocalDataSize() const override final { return 1; }

  T getSourceDistributionPower() const override final { return cosineExponent; }

  std::vector<std::string> getLocalDataLabels() const override final {
    return {dataLabel};
  }

private:
  const T stickingProbability = 1.;
  const T cosineExponent = 1.;
  const std::string dataLabel = "flux";
};

// psVelocityField
class PyVelocityField : public psVelocityField<T> {
  using psVelocityField<T>::psVelocityField;

public:
  T getScalarVelocity(const std::array<T, 3> &coordinate, int material,
                      const std::array<T, 3> &normalVector,
                      unsigned long pointId) override {
    PYBIND11_OVERRIDE(T, psVelocityField<T>, getScalarVelocity, coordinate,
                      material, normalVector, pointId);
  }
  // if we declare a typedef for std::array<T,3>, we will no longer get this
  // error: the compiler doesn't understand why std::array gets 2 template
  // arguments
  // add template argument as the preprocessor becomes confused with the comma
  // in std::array<T, 3>
  typedef std::array<T, 3> arrayType;
  std::array<T, 3> getVectorVelocity(const std::array<T, 3> &coordinate,
                                     int material,
                                     const std::array<T, 3> &normalVector,
                                     unsigned long pointId) override {
    PYBIND11_OVERRIDE(
        arrayType, // add template argument here, as the preprocessor becomes
                   // confused with the comma in std::array<T, 3>
        psVelocityField<T>, getVectorVelocity, coordinate, material,
        normalVector, pointId);
  }

  T getDissipationAlpha(int direction, int material,
                        const std::array<T, 3> &centralDifferences) override {
    PYBIND11_OVERRIDE(T, psVelocityField<T>, getDissipationAlpha, direction,
                      material, centralDifferences);
  }
  void setVelocities(psSmartPointer<std::vector<T>> passedVelocities) override {
    PYBIND11_OVERRIDE(void, psVelocityField<T>, setVelocities,
                      passedVelocities);
  }
  int getTranslationFieldOptions() const override {
    PYBIND11_OVERRIDE(int, psVelocityField<T>, getTranslationFieldOptions, );
  }
};

// a function to declare GeometricDistributionModel of type DistType
template <typename NumericType, int D, typename DistType>
void declare_GeometricDistributionModel(pybind11::module &m,
                                        const std::string &typestr) {
  using Class = GeometricDistributionModel<NumericType, D, DistType>;

  pybind11::class_<Class, psSmartPointer<Class>>(m, typestr.c_str())
      .def(pybind11::init<psSmartPointer<DistType>>(), pybind11::arg("dist"))
      .def(pybind11::init<psSmartPointer<DistType>,
                          psSmartPointer<lsDomain<NumericType, D>>>(),
           pybind11::arg("dist"), pybind11::arg("mask"))
      .def("apply", &Class::apply);
}