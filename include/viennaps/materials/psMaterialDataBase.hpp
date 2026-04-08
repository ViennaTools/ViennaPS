#pragma once

#include "psMaterialDBEntry.hpp"
#include "psMaterialValueMap.hpp"

namespace viennaps {

class MaterialDataBase {
public:
  using Entry = materials::DBEntry;

private:
  MaterialValueMap<Entry> materialData_;
};

} // namespace viennaps