#pragma once

#include "psImplantConstants.hpp"
#include <csImplantModel.hpp>

#include <algorithm>
#include <cmath>
#include <functional>

namespace viennaps {

using viennacore::SmartPointer;

template <typename NumericType, int D>
using ImplantModel = viennacs::ImplantModel<NumericType, D>;

namespace impl {

template <typename NumericType>
inline NumericType smoothstep(const NumericType edge0, const NumericType edge1,
                              const NumericType x) {
  if (edge1 <= edge0)
    return x >= edge0 ? NumericType(1) : NumericType(0);

  const auto t = std::clamp((x - edge0) / (edge1 - edge0), NumericType(0),
                            NumericType(1));
  return t * t * (NumericType(3) - NumericType(2) * t);
}

template <typename NumericType>
inline NumericType integrateTrapezoidal(
    const std::function<NumericType(NumericType)> &func, NumericType start,
    NumericType stop, NumericType step) {
  if (stop <= start || step <= NumericType(0))
    return NumericType(0);

  NumericType integral = 0;
  NumericType x0 = start;
  NumericType y0 = func(x0);
  for (NumericType x1 = start + step; x1 <= stop + step / 2; x1 += step) {
    const auto clippedX1 = std::min(x1, stop);
    const auto y1 = func(clippedX1);
    integral += (y0 + y1) * (clippedX1 - x0) / NumericType(2);
    x0 = clippedX1;
    y0 = y1;
    if (clippedX1 >= stop)
      break;
  }
  return integral;
}

} // namespace impl

enum class LateralStraggleModel {
  Constant,
  ExponentialDepthDecay,
  LinearDepthScale,
  LogSumExpDepthScale,
};

template <typename NumericType> struct LateralStraggleParameters {
  LateralStraggleModel model = LateralStraggleModel::Constant;
  NumericType mu = NumericType(0);
  NumericType sigma = NumericType(1);
  NumericType scale = NumericType(1);

  // Exponential depth decay: sigma_l(x) = sigma * exp(-x / (sigma * lv))
  NumericType lv = NumericType(1);

  // Linear depth scaling: sigma_l(x) = sigma * (1 + deltaSigma * (x / Rp - 1))
  NumericType deltaSigma = NumericType(0);
  NumericType referenceRange = NumericType(0);

  // Smooth two-branch log-sum-exp depth scaling coefficients.
  NumericType p1 = NumericType(0);
  NumericType p2 = NumericType(0);
  NumericType p3 = NumericType(0);
  NumericType p4 = NumericType(0);
  NumericType p5 = NumericType(0);
};

namespace impl {

template <typename NumericType>
inline NumericType computeLateralSigma(
    NumericType depth, const LateralStraggleParameters<NumericType> &params,
    NumericType verticalSigma) {
  const auto safeScale = std::max(params.scale, NumericType(0));
  const auto safeSigma = std::max(params.sigma, NumericType(1e-9));
  const auto clampedDepth = std::max(depth, NumericType(0));

  NumericType sigma = safeSigma;
  switch (params.model) {
  case LateralStraggleModel::Constant:
    sigma = safeSigma;
    break;
  case LateralStraggleModel::ExponentialDepthDecay: {
    const auto safeLv = std::max(params.lv, NumericType(1e-9));
    sigma = safeSigma *
            std::exp(-clampedDepth / (safeSigma * safeLv));
    break;
  }
  case LateralStraggleModel::LinearDepthScale: {
    const auto rp = std::max(params.referenceRange, NumericType(1e-9));
    sigma = safeSigma *
            (NumericType(1) + params.deltaSigma * (clampedDepth / rp - NumericType(1)));
    break;
  }
  case LateralStraggleModel::LogSumExpDepthScale: {
    const auto rp = std::max(params.referenceRange, NumericType(1e-9));
    const auto safeP1 = std::max(std::abs(params.p1), NumericType(1e-9));
    const auto arg1 =
        safeP1 * (params.p2 * clampedDepth / rp + params.p3);
    const auto arg2 =
        safeP1 * (params.p4 * clampedDepth / rp + params.p5);
    const auto shape =
        std::log(std::exp(arg1) + std::exp(arg2)) / safeP1;
    sigma = std::max(NumericType(0.01), shape) *
            std::max(verticalSigma, NumericType(1e-9));
    break;
  }
  }

  sigma *= safeScale;
  if (!std::isfinite(sigma) || sigma <= NumericType(0))
    return NumericType(1e-9);
  return sigma;
}

template <typename NumericType>
inline NumericType computeMaxLateralSigma(
    NumericType maxDepth, const LateralStraggleParameters<NumericType> &params,
    NumericType verticalSigma) {
  NumericType maxSigma = NumericType(0);
  constexpr int sampleCount = 128;
  for (int i = 0; i <= sampleCount; ++i) {
    const auto depth =
        maxDepth * static_cast<NumericType>(i) / static_cast<NumericType>(sampleCount);
    maxSigma = std::max(maxSigma,
                        computeLateralSigma(depth, params, verticalSigma));
  }
  return maxSigma;
}

template <typename NumericType>
inline NumericType gaussianLateralProfile(
    NumericType offset, NumericType depth,
    const LateralStraggleParameters<NumericType> &params,
    NumericType verticalSigma) {
  const auto sigma = computeLateralSigma(depth, params, verticalSigma);
  return (NumericType(1) / (sigma * std::sqrt(2 * M_PI))) *
         std::exp(-NumericType(0.5) *
                  std::pow((offset - params.mu) / sigma, 2));
}

} // namespace impl

template <class NumericType, int D>
class ImplantPearsonIV final : public ImplantModel<NumericType, D> {
public:
  ImplantPearsonIV(const constants::PearsonIVParameters<NumericType> &params,
                   NumericType lateralMu, NumericType lateralSigma)
      : ImplantPearsonIV(params, LateralStraggleParameters<NumericType>{
                                     LateralStraggleModel::Constant, lateralMu,
                                     lateralSigma}) {}

  ImplantPearsonIV(
      const constants::PearsonIVParameters<NumericType> &params,
      const LateralStraggleParameters<NumericType> &lateralParams)
      : params_(params), lateralParams_(lateralParams),
        maxDepth_(std::max(params.mu + NumericType(8) * params.sigma,
                           NumericType(0))) {
    const auto integrationStep =
        std::max(params_.sigma / NumericType(50), NumericType(1e-3));
    depthNormalization_ =
        impl::integrateTrapezoidal<NumericType>(
            [this](NumericType depth) { return rawDepthProfile(depth); },
            NumericType(0), maxDepth_, integrationStep);
    if (depthNormalization_ <= NumericType(0))
      depthNormalization_ = NumericType(1);

    const auto maxSigma =
        impl::computeMaxLateralSigma(maxDepth_, lateralParams_, params_.sigma);
    maxLateralRange_ = std::abs(lateralParams_.mu) + NumericType(6) * maxSigma;
  }

  NumericType getDepthProfile(NumericType depth) override {
    return rawDepthProfile(depth) / depthNormalization_;
  }

  NumericType getLateralProfile(NumericType offset,
                                NumericType depth) override {
    return impl::gaussianLateralProfile(offset, depth, lateralParams_,
                                        params_.sigma);
  }

  NumericType getMaxDepth() override { return maxDepth_; }

  NumericType getMaxLateralRange() override { return maxLateralRange_; }

protected:
  NumericType rawDepthProfile(NumericType depth) const {
    if (depth < NumericType(0))
      return NumericType(0);
    return constants::PearsonIV(depth, params_);
  }

private:
  const constants::PearsonIVParameters<NumericType> params_;
  const LateralStraggleParameters<NumericType> lateralParams_;
  NumericType depthNormalization_ = NumericType(1);
  NumericType maxDepth_ = NumericType(0);
  NumericType maxLateralRange_ = NumericType(0);
};

template <class NumericType, int D>
class ImplantPearsonIVChanneling final : public ImplantModel<NumericType, D> {
public:
  ImplantPearsonIVChanneling(
      const constants::PearsonIVParameters<NumericType> &params,
      NumericType lateralMu, NumericType lateralSigma,
      NumericType tailFraction, NumericType tailStartDepth,
      NumericType tailDecayLength, NumericType tailBlendWidth = NumericType(0))
      : ImplantPearsonIVChanneling(
            params,
            LateralStraggleParameters<NumericType>{LateralStraggleModel::Constant,
                                                   lateralMu, lateralSigma},
            tailFraction, tailStartDepth, tailDecayLength, tailBlendWidth) {}

  ImplantPearsonIVChanneling(
      const constants::PearsonIVParameters<NumericType> &params,
      const LateralStraggleParameters<NumericType> &lateralParams,
      NumericType tailFraction, NumericType tailStartDepth,
      NumericType tailDecayLength, NumericType tailBlendWidth = NumericType(0))
      : randomImplant_(params, lateralParams), lateralParams_(lateralParams),
        primarySigma_(params.sigma),
        tailFraction_(std::clamp(tailFraction, NumericType(0), NumericType(1))),
        tailStartDepth_(tailStartDepth), tailDecayLength_(tailDecayLength),
        tailBlendWidth_(tailBlendWidth),
        maxDepth_(std::max(randomImplant_.getMaxDepth(),
                           tailStartDepth + NumericType(10) * tailDecayLength)) {
    const auto integrationStep =
        std::max(params.sigma / NumericType(50), NumericType(1e-3));
    tailNormalization_ = impl::integrateTrapezoidal<NumericType>(
        [this](NumericType depth) { return rawTailProfile(depth); },
        NumericType(0), maxDepth_, integrationStep);
    if (tailNormalization_ <= NumericType(0))
      tailNormalization_ = NumericType(1);

    const auto maxSigma =
        impl::computeMaxLateralSigma(maxDepth_, lateralParams_, params.sigma);
    maxLateralRange_ = std::abs(lateralParams_.mu) + NumericType(6) * maxSigma;
  }

  NumericType getDepthProfile(NumericType depth) override {
    const auto randomDensity = randomImplant_.getDepthProfile(depth);
    if (tailFraction_ <= NumericType(0) || tailDecayLength_ <= NumericType(0))
      return randomDensity;

    const auto tailDensity = rawTailProfile(depth) / tailNormalization_;
    return (NumericType(1) - tailFraction_) * randomDensity +
           tailFraction_ * tailDensity;
  }

  NumericType getLateralProfile(NumericType offset,
                                NumericType depth) override {
    return impl::gaussianLateralProfile(offset, depth, lateralParams_,
                                        primarySigma_);
  }

  NumericType getMaxDepth() override { return maxDepth_; }

  NumericType getMaxLateralRange() override { return maxLateralRange_; }

private:
  NumericType rawTailProfile(NumericType depth) const {
    if (depth < NumericType(0) || tailDecayLength_ <= NumericType(0))
      return NumericType(0);
    const auto onset = impl::smoothstep(
        tailStartDepth_ - NumericType(0.5) * tailBlendWidth_,
        tailStartDepth_ + NumericType(0.5) * tailBlendWidth_, depth);
    if (onset <= NumericType(0))
      return NumericType(0);
    return onset *
           std::exp(-(depth - tailStartDepth_) / tailDecayLength_);
  }

  ImplantPearsonIV<NumericType, D> randomImplant_;
  LateralStraggleParameters<NumericType> lateralParams_;
  NumericType primarySigma_;
  NumericType tailFraction_;
  NumericType tailStartDepth_;
  NumericType tailDecayLength_;
  NumericType tailBlendWidth_;
  NumericType tailNormalization_ = NumericType(1);
  NumericType maxDepth_ = NumericType(0);
  NumericType maxLateralRange_ = NumericType(0);
};

template <class NumericType, int D>
class ImplantDualPearsonIV final : public ImplantModel<NumericType, D> {
public:
  ImplantDualPearsonIV(
      const constants::PearsonIVParameters<NumericType> &headParams,
      const constants::PearsonIVParameters<NumericType> &tailParams,
      NumericType headFraction, NumericType headLateralMu,
      NumericType headLateralSigma, NumericType tailLateralMu,
      NumericType tailLateralSigma)
      : ImplantDualPearsonIV(
            headParams, tailParams, headFraction,
            LateralStraggleParameters<NumericType>{LateralStraggleModel::Constant,
                                                   headLateralMu, headLateralSigma},
            LateralStraggleParameters<NumericType>{LateralStraggleModel::Constant,
                                                   tailLateralMu, tailLateralSigma}) {}

  ImplantDualPearsonIV(
      const constants::PearsonIVParameters<NumericType> &headParams,
      const constants::PearsonIVParameters<NumericType> &tailParams,
      NumericType headFraction,
      const LateralStraggleParameters<NumericType> &headLateralParams,
      const LateralStraggleParameters<NumericType> &tailLateralParams)
      : headImplant_(headParams, headLateralParams),
        tailImplant_(tailParams, tailLateralParams),
        headFraction_(
            std::clamp(headFraction, NumericType(0), NumericType(1))),
        maxDepth_(std::max(headImplant_.getMaxDepth(),
                           tailImplant_.getMaxDepth())),
        maxLateralRange_(
            std::max(headImplant_.getMaxLateralRange(),
                     tailImplant_.getMaxLateralRange())) {}

  NumericType getDepthProfile(NumericType depth) override {
    return headFraction_ * headImplant_.getDepthProfile(depth) +
           (NumericType(1) - headFraction_) *
               tailImplant_.getDepthProfile(depth);
  }

  NumericType getProfile(NumericType depth, NumericType offset) override {
    return headFraction_ * headImplant_.getProfile(depth, offset) +
           (NumericType(1) - headFraction_) *
               tailImplant_.getProfile(depth, offset);
  }

  NumericType getLateralProfile(NumericType offset,
                                NumericType depth) override {
    return headFraction_ * headImplant_.getLateralProfile(offset, depth) +
           (NumericType(1) - headFraction_) *
               tailImplant_.getLateralProfile(offset, depth);
  }

  NumericType getMaxDepth() override { return maxDepth_; }

  NumericType getMaxLateralRange() override { return maxLateralRange_; }

private:
  ImplantPearsonIV<NumericType, D> headImplant_;
  ImplantPearsonIV<NumericType, D> tailImplant_;
  NumericType headFraction_;
  NumericType maxDepth_ = NumericType(0);
  NumericType maxLateralRange_ = NumericType(0);
};

} // namespace viennaps
