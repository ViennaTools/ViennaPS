#pragma once

#include "psProcessContext.hpp"
#include <memory>

#define DEFINE_CLASS_NAME(CLASS)                                               \
  static constexpr std::string_view kName = #CLASS;                            \
  std::string_view name() const noexcept override { return kName; }

namespace viennaps {

template <typename NumericType, int D> class ProcessStrategy {
public:
  virtual ~ProcessStrategy() = default;

  virtual ProcessResult execute(ProcessContext<NumericType, D> &context) = 0;
  virtual bool
  canHandle(const ProcessContext<NumericType, D> &context) const = 0;
  virtual std::string_view name() const = 0;
};

} // namespace viennaps