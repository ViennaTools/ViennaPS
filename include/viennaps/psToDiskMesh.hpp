#pragma once

#include "psDomain.hpp"

#include <lsToDiskMesh.hpp>

namespace viennaps {

using namespace viennacore;

template <class NumericType, int D> class ToDiskMesh {
  using translatorType =
      SmartPointer<std::unordered_map<unsigned long, unsigned long>>;
  using psDomainType = SmartPointer<Domain<NumericType, D>>;
  using meshType = SmartPointer<lsMesh<NumericType>>;

  psDomainType domain;
  translatorType translator;
  meshType mesh;

public:
  ToDiskMesh() {}

  ToDiskMesh(psDomainType passedDomain, meshType passedMesh)
      : domain(passedDomain), mesh(passedMesh) {}

  ToDiskMesh(psDomainType passedDomain, meshType passedMesh,
             translatorType passedTranslator)
      : domain(passedDomain), mesh(passedMesh), translator(passedTranslator) {}

  void setDomain(psDomainType passedDomain) { domain = passedDomain; }

  void setMesh(meshType passedMesh) { mesh = passedMesh; }

  void setTranslator(translatorType passedTranslator) {
    translator = passedTranslator;
  }

  translatorType getTranslator() const { return translator; }

  void apply() {
    lsToDiskMesh<NumericType, D> meshConverter;
    meshConverter.setMesh(mesh);
    if (domain->getMaterialMap())
      meshConverter.setMaterialMap(domain->getMaterialMap()->getMaterialMap());
    if (translator.get())
      meshConverter.setTranslator(translator);
    for (const auto ls : *domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(ls);
    }
    meshConverter.apply();
  }
};

} // namespace viennaps
