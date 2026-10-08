#pragma once

// The binary-cell PMC on the GPU: the parameter block the host driver
// (psVoxelPMCGPU.hpp) uploads and the device programs
// (gpu/models/VoxelPMC.cuh) read. ONE definition, included by both, so the
// two sides cannot drift apart.
//
// The device decides, per hit, what the CPU decides per hit: the re-hit
// rule, the spreading of the reaction site, stick or reflect from the cell's
// state, the restart one cell out along the normal. The cell states live in
// device memory and the hit programs change them as the CPU does, so a
// particle sees what the particles before it left, in whatever order the
// device runs them. Each reaction is also recorded as an event, which the
// host applies with the CPU's bookkeeping: cascade steps, SiF4 and ion
// removal, settling. The traced band is built once per step, as the CPU's
// embree band is.

namespace viennaps {

/// A band cell's live state. 0 to 2 are the CPU's own codes.
enum VoxelPMCStateGPU : int {
  pmcSi = 0,   ///< bare silicon
  pmcSiF = 1,  ///< fluorinated, the cascade step in bandStep
  pmcSiO = 2,  ///< oxidised
  pmcMask = 3, ///< reacts with nothing, reflects
  pmcGone = 4, ///< removed earlier in this step
  pmcBuried = 5 ///< solid with gas only at a corner: in the CPU's embree
                ///< band, and a ray that lands in it is dropped, as there
};

struct VoxelPMCParamsGPU {
  static constexpr int maxSteps = 8;
  static constexpr int numCounters = 19;

  // ---- the lattice and the band (exposed solid cells, mask included)
  int D = 2;
  int dims[3] = {1, 1, 1};
  int bandSize = 0;
  const int *bandIdx = nullptr;       ///< band -> lattice index, 3 ints each
  int *bandState = nullptr;           ///< live VoxelPMCStateGPU per cell
  int *bandStep = nullptr;            ///< live cascade step (F bonded)
  const float *fitNormal = nullptr;   ///< band -> 3 floats, plane fit at
                                      ///< reflectRadius (re-hit rule, ion)
  const float *walkOut = nullptr;     ///< band -> restart distance, < 0 none
  const unsigned char *normalValid = nullptr; ///< the estimator gave a normal
  // spreading: candidate reaction sites per band cell, CSR
  const int *hopStart = nullptr;      ///< bandSize + 1 offsets
  const int *hopList = nullptr;       ///< band ids (exposed, not mask)
  const unsigned char *hopCheb = nullptr; ///< Chebyshev distance of each

  // ---- spreading of the reaction site
  int hopR = 2;
  int hopWeighted = 1;                ///< 0: uniform over the box
  float hopW[3] = {0.30f, 0.20f, 0.15f};
  int fHop = 1;
  int oHop = 1;

  // ---- the fluorination cascade and oxygen
  int cascadeTop = 3;
  float cascadeP[maxSteps] = {};
  float stickO = 1.f;

  // ---- the re-hit rule
  int sameFacet = 1;
  int sameFacetCells = 2;
  float cosLim = 0.8660254f;
  int maxBounce = 200;

  // ---- what is being launched: 0 F, 1 O (the ion has its own programs)
  int species = 0;

  // ---- output
  int *events = nullptr;              ///< neutral: reaction site (band id)
  float *ionEvents = nullptr;         ///< ion: (band id, E, cos, reflected)
  unsigned int *eventCount = nullptr;
  unsigned int eventCapacity = 0;
  unsigned long long *counters = nullptr; ///< numCounters entries

  // ---- the ion
  float meanEnergy = 100.f, sigmaEnergy = 10.f;
  float inflectAngle = 1.5f, n_l = 10.f, minAngle = 1.4f; // radians
  float thetaRMin = 1.22f, thetaRMax = 1.57f;              // radians
  float minEth = 0.f;
};

/// Device counters, summed into the PMC's own on the host.
enum VoxelPMCCounterGPU : int {
  pmcHits = 0,      ///< surface hits (every hit, refused ones included)
  pmcRefused,       ///< re-hits refused as sub-resolution
  pmcCornerKeep,    ///< close re-hits kept, being a real corner
  pmcMaskReflect,   ///< hits on the mask, which reacts with nothing
  pmcFArrivals,     ///< F hits on silicon tested for a reaction
  pmcFonO,          ///< F reflected off SiO
  pmcStickFail,     ///< sticking test failed, reflects
  pmcOBare,         ///< O found its reaction site bare
  pmcOOccupied,     ///< O found it taken
  pmcHopMoved,      ///< bond placed on a neighbour
  pmcHopStay,       ///< bond kept on the hit cell
  pmcBounceCap,     ///< bounce budget spent
  pmcNoEscape,      ///< restart could not reach gas, dropped
  pmcIonImpacts,    ///< ion impacts on non-mask cells (events)
  pmcIonBounce,     ///< ion reflections
  pmcOverflow,      ///< events beyond the buffer, lost
  pmcGoneHit,       ///< hits on a cell removed earlier in the step, lost
  pmcBuriedHit,     ///< hits on a buried band cell, dropped
  pmcNoNormalHit    ///< hits on a cell the estimator gave no normal
};

} // namespace viennaps
