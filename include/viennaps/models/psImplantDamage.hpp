#pragma once

#include <csImplantModel.hpp>
#include "psImplantPearson.hpp"

#include <algorithm>
#include <cmath>
#include <functional>

namespace viennaps {

template <class NumericType, int D>
class ImplantDamageHobler final : public ImplantModel<NumericType, D> {
public:
  ImplantDamageHobler(
      NumericType projectedRange, NumericType verticalSigma,
      NumericType lambda, NumericType defectsPerIon, NumericType lateralSigma,
      NumericType lateralDeltaSigma = NumericType(0))
      : ImplantDamageHobler(projectedRange, verticalSigma, lambda,
                            defectsPerIon,
                            LateralStraggleParameters<NumericType>{
                                LateralStraggleModel::LinearDepthScale, NumericType(0),
                                lateralSigma, NumericType(1), NumericType(1),
                                lateralDeltaSigma, projectedRange}) {}

  ImplantDamageHobler(
      NumericType projectedRange, NumericType verticalSigma,
      NumericType lambda, NumericType defectsPerIon,
      const LateralStraggleParameters<NumericType> &lateralParams)
      : rp_(std::max(projectedRange, NumericType(0))),
        sigma_(std::max(verticalSigma, NumericType(1e-9))),
        lambda_(lambda),
        defectsPerIon_(std::max(defectsPerIon, NumericType(0))),
        lateralParams_(lateralParams) {
    maxDepth_ = std::max(rp_ + NumericType(8) * sigma_, NumericType(0));
    const auto lambdaAbs = std::abs(lambda_);
    if (lambdaAbs > NumericType(1e-9)) {
      const auto x0 =
          lambda_ > NumericType(0)
              ? rp_ - (sigma_ * sigma_) / lambdaAbs
              : rp_ + (sigma_ * sigma_) / lambdaAbs;
      maxDepth_ = std::max(maxDepth_, x0 + NumericType(10) * lambdaAbs);
    }

    const auto integrationStep =
        std::max(sigma_ / NumericType(50), NumericType(1e-3));
    depthNormalization_ =
        impl::integrateTrapezoidal<NumericType>(
            [this](NumericType depth) { return rawDepthShape(depth); },
            NumericType(0), maxDepth_, integrationStep);
    if (depthNormalization_ <= NumericType(0))
      depthNormalization_ = NumericType(1);

    const auto maxSigma =
        impl::computeMaxLateralSigma(maxDepth_, lateralParams_, sigma_);
    maxLateralRange_ = std::abs(lateralParams_.mu) + NumericType(6) * maxSigma;
  }

  NumericType getDepthProfile(NumericType depth) override {
    return defectsPerIon_ * rawDepthShape(depth) / depthNormalization_;
  }

  NumericType getLateralProfile(NumericType offset,
                                NumericType depth) override {
    return impl::gaussianLateralProfile(offset, depth, lateralParams_, sigma_);
  }

  NumericType getMaxDepth() override { return maxDepth_; }

  NumericType getMaxLateralRange() override { return maxLateralRange_; }

private:
  NumericType gaussianShape(NumericType depth) const {
    return std::exp(-NumericType(0.5) *
                    std::pow((depth - rp_) / sigma_, NumericType(2)));
  }

  NumericType rawDepthShape(NumericType depth) const {
    if (depth < NumericType(0))
      return NumericType(0);

    const auto lambdaAbs = std::abs(lambda_);
    if (lambdaAbs <= NumericType(1e-9))
      return gaussianShape(depth);

    if (lambda_ > NumericType(0)) {
      // Type-1 Hobler: exponential near surface, Gaussian near projected range.
      const auto x0 = rp_ - (sigma_ * sigma_) / lambdaAbs;
      if (depth > x0)
        return gaussianShape(depth);
      const auto match = gaussianShape(x0);
      return match * std::exp((depth - x0) / lambdaAbs);
    }

    // Type-2 Hobler: Gaussian before Rp, exponential tail into the bulk.
    const auto x0 = rp_ + (sigma_ * sigma_) / lambdaAbs;
    if (depth <= x0)
      return gaussianShape(depth);
    const auto match = gaussianShape(x0);
    return match * std::exp(-(depth - x0) / lambdaAbs);
  }

  NumericType rp_ = NumericType(0);
  NumericType sigma_ = NumericType(1);
  NumericType lambda_ = NumericType(0);
  NumericType defectsPerIon_ = NumericType(0);
  LateralStraggleParameters<NumericType> lateralParams_;
  NumericType depthNormalization_ = NumericType(1);
  NumericType maxDepth_ = NumericType(0);
  NumericType maxLateralRange_ = NumericType(0);
};

} // namespace viennaps
