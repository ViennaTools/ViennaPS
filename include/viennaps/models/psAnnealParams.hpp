#pragma once

#include "psAnneal.hpp"

namespace viennaps {

template <typename NumericType> struct AnnealParams {
  bool useConstantDiffusionCoefficient = false;
  NumericType diffusionCoefficient = 0;
  NumericType dopantD0 = 0;
  NumericType dopantEa_eV = 0;

  NumericType defectSourceHistoryWeight = 0;
  NumericType defectSourceLastDamageWeight = 1;

  NumericType interstitialDiffusivity = 0;
  NumericType vacancyDiffusivity = 0;

  bool enableDefectEquilibrium = false;
  NumericType interstitialEqC0 = 0;
  NumericType interstitialEqEa_eV = 0;
  NumericType vacancyEqC0 = 0;
  NumericType vacancyEqEa_eV = 0;

  NumericType defectRecombinationRate = 0;
  NumericType interstitialSinkRate = 0;
  NumericType vacancySinkRate = 0;

  NumericType scoreIFactor = 0.5;
  NumericType scoreVFactor = 0.5;
  NumericType scoreDFactor = 1.0;

  bool enableTedFromScoreDFactor = false;
  NumericType tedCoefficient = 0;
  NumericType tedCoefficientScale = 0.5;
  NumericType tedNormalization = 1e20;

  bool enableDefectClustering = false;
  NumericType clusterKfi = 0;
  NumericType clusterKfc = 0;
  NumericType clusterKr = 0;
  NumericType clusterInitFraction = 0;

  bool enableSolidActivation = true;
  NumericType solidSolubilityC0 = 0;
  NumericType solidSolubilityEa_eV = 0;
};

template <typename NumericType, int D>
inline void applyAnnealParams(Anneal<NumericType, D> &anneal,
                              const AnnealParams<NumericType> &p) {
  if (p.useConstantDiffusionCoefficient)
    anneal.setDiffusionCoefficient(p.diffusionCoefficient);
  else
    anneal.setArrheniusParameters(p.dopantD0, p.dopantEa_eV);
  anneal.setDefectSourceWeights(p.defectSourceHistoryWeight,
                                p.defectSourceLastDamageWeight);
  anneal.setDefectDiffusivities(p.interstitialDiffusivity,
                                p.vacancyDiffusivity);
  anneal.enableDefectEquilibrium(p.enableDefectEquilibrium);
  if (p.enableDefectEquilibrium)
    anneal.setDefectEquilibriumArrhenius(p.interstitialEqC0,
                                         p.interstitialEqEa_eV, p.vacancyEqC0,
                                         p.vacancyEqEa_eV);
  anneal.setDefectReactionRates(p.defectRecombinationRate,
                                p.interstitialSinkRate, p.vacancySinkRate);
  const auto i =
      p.scoreIFactor > NumericType(0) ? p.scoreIFactor : NumericType(0);
  const auto v =
      p.scoreVFactor > NumericType(0) ? p.scoreVFactor : NumericType(0);
  if (i + v > NumericType(0))
    anneal.setDefectPartition(i / (i + v), v / (i + v));
  if (p.enableSolidActivation && p.solidSolubilityC0 > 0) {
    anneal.enableSolidActivation(true);
    anneal.setSolidSolubilityArrhenius(p.solidSolubilityC0,
                                       p.solidSolubilityEa_eV);
  }
  if (p.enableTedFromScoreDFactor)
    anneal.setDefectEnhancedDiffusionFromDamageFactor(
        p.scoreDFactor, p.tedCoefficientScale, p.tedNormalization);
  else if (p.tedCoefficient > NumericType(0))
    anneal.setDefectEnhancedDiffusion(p.tedCoefficient, p.tedNormalization);
  if (p.enableDefectClustering) {
    anneal.enableDefectClustering(true);
    anneal.setDefectClusterKinetics(p.clusterKfi, p.clusterKfc, p.clusterKr);
    anneal.setDefectClusterInitFraction(p.clusterInitFraction);
  }
}

} // namespace viennaps
