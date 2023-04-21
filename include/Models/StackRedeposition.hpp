#pragma once

#include <csDenseCellSet.hpp>

#include <lsToDiskMesh.hpp>

#include <psAdvectionCallback.hpp>
#include <psDomain.hpp>

// The selective etching model works in accordance with the geometry generated
// by psMakeStack
template <class NumericType>
class SelectiveEtchingVelocityField : public psVelocityField<NumericType> {
public:
  SelectiveEtchingVelocityField(const NumericType pRate,
                                const NumericType pOxideRate,
                                const int pDepoMat)
      : rate(pRate), oxide_rate(pOxideRate), depoMat(pDepoMat) {}

  NumericType getScalarVelocity(const std::array<NumericType, 3> &coordinate,
                                int matId,
                                const std::array<NumericType, 3> &normalVector,
                                unsigned long pointId) override {
    if (matId == 0 || matId == depoMat) {
      return 0.;
    }

    if (matId % 2 == 0) {
      return -rate;
    } else {
      return -oxide_rate;
    }
  }

  bool useTranslationField() const override { return false; }

private:
  const NumericType rate;
  const NumericType oxide_rate;
  const int depoMat;
};