#pragma once

#include <lsToDiskMesh.hpp>

#include <psDomain.hpp>

template <class NumericType, int D> class psToDiskMesh {
  using translatorType =
      psSmartPointer<std::unordered_map<unsigned long, unsigned long>>;
  using psDomainType = psSmartPointer<psDomain<NumericType, D>>;
  using meshType = psSmartPointer<lsMesh<NumericType>>;

  psDomainType domain;
  translatorType translator;
  meshType mesh;

public:
  psToDiskMesh() {}

  psToDiskMesh(psDomainType passedDomain, meshType passedMesh)
      : domain(passedDomain), mesh(passedMesh) {}

  psToDiskMesh(psDomainType passedDomain, meshType passedMesh,
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
    meshConverter.setMaterialMap(domain->getMaterialMap()->getMaterialMap());
    if (translator.get())
      meshConverter.setTranslator(translator);
    for (const auto ls : *domain->getLevelSets()) {
      meshConverter.insertNextLevelSet(ls);
    }
    meshConverter.apply();
  }
};