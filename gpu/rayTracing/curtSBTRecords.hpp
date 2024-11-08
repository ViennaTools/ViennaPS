#pragma once

#include <optix_types.h>
#include <utGDT.hpp>

struct HitSBTData {
  gdt::vec3f *vertex;
  gdt::vec3i *index;
  bool isBoundary;
  void *cellData;
};

struct HitSBTDiskData {
  gdt::vec3f *vertex;
  gdt::vec3i *index;
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