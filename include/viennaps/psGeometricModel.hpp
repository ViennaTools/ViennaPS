// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2025 TU Wien

#pragma once

#include "psDomain.hpp"

#include <vcSmartPointer.hpp>

namespace viennaps {

using namespace viennacore;

template <typename NumericType, int D> class GeometricModel {
protected:
  SmartPointer<Domain<NumericType, D>> domain = nullptr;
  std::unordered_map<std::string, std::vector<NumericType>> processData;

public:
  virtual ~GeometricModel() = default;

  void setDomain(SmartPointer<Domain<NumericType, D>> passedDomain) {
    domain = passedDomain;
  }

  virtual void apply() {}
};

} // namespace viennaps
