#pragma once

#include <optix_types.h>
#include <vcVectorUtil.hpp>

namespace viennaps {

namespace gpu {

struct HitSBTData {
  viennacore::Vec3Df *vertex;
  viennacore::Vec3D<unsigned> *index;
  bool isBoundary;
  void *cellData;
};

struct HitSBTDiskData {
  viennacore::Vec3Df *vertex;
  viennacore::Vec3D<unsigned> *index;
  bool isBoundary;
  void *cellData;
};

// SBT record for a raygen program
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

// SBT record for a miss program
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data; // dummy value
};

// SBT record for a hitgroup program
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  HitSBTData data;
};

// SBT record for a hitgroup program
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecordDisk {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  HitSBTDiskData data;
};

} // namespace gpu
} // namespace viennaps