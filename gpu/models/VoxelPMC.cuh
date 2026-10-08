#pragma once

// The binary-cell PMC on the GPU (psVoxelPMC, cascade path): the per-hit
// decisions the CPU makes in deliver() and in the ion's own path. Each band
// cell's state lives in device memory, and the hit programs change it as the
// CPU does: a cascade step, SiO on bare silicon, a cell gone with its SiF4.
// A particle therefore sees what the particles before it left. Every reaction
// is also recorded as an event for the host, which applies it with the CPU's
// bookkeeping. Anything that does not react re-emits exactly as on the CPU.
// The analog rule is kept: a particle either reacts (weight 0, the ray ends)
// or reflects with its full weight, never a fraction.

#include "vcContext.hpp"
#include "vcVectorType.hpp"

#include "raygLaunchParams.hpp"
#include "raygReflection.hpp"

#include "models/psPipelineParameters.hpp"
#include "models/psVoxelPMCParamsGPU.hpp"

extern "C" __constant__ viennaray::gpu::LaunchParams launchParams;

__forceinline__ __device__ const viennaps::VoxelPMCParamsGPU *pmcParams() {
  return reinterpret_cast<const viennaps::VoxelPMCParamsGPU *>(
      launchParams.customData);
}

__forceinline__ __device__ void pmcCount(const viennaps::VoxelPMCParamsGPU *p,
                                         int which) {
  atomicAdd(&p->counters[which], 1ull);
}

/// A live state or step, read past L1 so a change another ray made in this
/// launch is seen.
__forceinline__ __device__ int pmcLoad(const int *a) { return __ldcg(a); }

/// The reaction site of a neutral that landed on `hit`: a random exposed,
/// non-mask cell within hopR cells (Chebyshev), weighted by distance, the hit
/// cell included -- psVoxelPMC::hopSite. The candidates come precomputed per
/// band cell from the host; a cell removed since is no longer one.
__forceinline__ __device__ int pmcHopSite(const viennaps::VoxelPMCParamsGPU *p,
                                          int hit, viennaray::gpu::PerRayData *prd) {
  int best = hit;
  float wsum = 0.f;
  int seen = 0;
  for (int k = p->hopStart[hit]; k < p->hopStart[hit + 1]; ++k) {
    const int c = p->hopList[k];
    if (pmcLoad(p->bandState + c) == viennaps::pmcGone)
      continue;
    if (p->hopWeighted) {
      const int cheb = p->hopCheb[k];
      const float wgt = cheb < 3 ? p->hopW[cheb] : 0.f;
      if (wgt <= 0.f)
        continue;
      wsum += wgt;
      if (viennacore::getNextRand(&prd->RNGstate) * wsum < wgt)
        best = c;
    } else {
      ++seen;
      if (viennacore::getNextRand(&prd->RNGstate) * seen < 1.f)
        best = c;
    }
  }
  return best;
}

__forceinline__ __device__ void pmcRecord(const viennaps::VoxelPMCParamsGPU *p,
                                          int site) {
  const unsigned int k = atomicAdd(p->eventCount, 1u);
  if (k < p->eventCapacity)
    p->events[k] = site;
  else
    pmcCount(p, viennaps::pmcOverflow);
}

/// Hits the CPU does not keep. A cell removed earlier in this step: the
/// CPU's ray passes through the emptied cell into material that is not in the
/// band it traces, and is lost. A buried cell, which touches the gas only at a
/// corner: the CPU drops a ray that lands in one ("went inside").
__forceinline__ __device__ bool pmcDropped(const viennaps::VoxelPMCParamsGPU *p,
                                           viennaray::gpu::PerRayData *prd,
                                           unsigned int hit) {
  const int st = pmcLoad(p->bandState + hit);
  if (st != viennaps::pmcGone && st != viennaps::pmcBuried)
    return false;
  pmcCount(p, st == viennaps::pmcGone ? viennaps::pmcGoneHit
                                      : viennaps::pmcBuriedHit);
  prd->rayWeight = 0.f;
  return true;
}

/// The re-hit rule: a re-emitted neutral landing within sameFacetCells of the
/// cell it left gets no new reaction, unless the fitted normals there differ
/// by more than the corner angle.
__forceinline__ __device__ bool pmcSubResolution(const viennaps::VoxelPMCParamsGPU *p,
                                                 unsigned int prev, unsigned int hit) {
  if (!p->sameFacet || prev == 0xFFFFFFFFu)
    return false;
  int cheb = 0;
  for (int d = 0; d < p->D; ++d) {
    const int dd = p->bandIdx[3 * prev + d] - p->bandIdx[3 * hit + d];
    cheb = max(cheb, dd < 0 ? -dd : dd);
  }
  if (cheb > p->sameFacetCells)
    return false;
  const float *na = p->fitNormal + 3 * prev;
  const float *nb = p->fitNormal + 3 * hit;
  float la = 0.f, lb = 0.f, dot = 0.f;
  for (int d = 0; d < p->D; ++d) {
    la += na[d] * na[d];
    lb += nb[d] * nb[d];
    dot += na[d] * nb[d];
  }
  if (la <= 1e-12f || lb <= 1e-12f)
    return false;
  if (dot / sqrtf(la * lb) >= p->cosLim) {
    pmcCount(p, viennaps::pmcRefused);
    return true;
  }
  pmcCount(p, viennaps::pmcCornerKeep);
  return false;
}

//
// --- neutrals (F and O)
//

__forceinline__ __device__ void
pmcNeutralInit(viennaray::gpu::PerRayData *prd) {
  prd->primIDs[1] = 0xFFFFFFFFu; // no previous hit yet
}

__forceinline__ __device__ void
pmcNeutralCollision(const void *, viennaray::gpu::PerRayData *prd) {
  const auto *p = pmcParams();
  const unsigned int hit = prd->primID;
  if (pmcDropped(p, prd, hit))
    return;
  pmcCount(p, viennaps::pmcHits);
  if (!p->normalValid[hit])
    pmcCount(p, viennaps::pmcNoNormalHit);
  const bool skip = pmcSubResolution(p, prd->primIDs[1], hit);
  prd->primIDs[1] = hit;
  if (skip)
    return; // same continuum facet: no reaction, it travels on

  if (p->bandState[hit] == viennaps::pmcMask) {
    pmcCount(p, viennaps::pmcMaskReflect); // the mask reacts with nothing
    return;
  }

  int site = static_cast<int>(hit);
  if (p->species == 1) {
    // O: SiO if the reaction site is bare silicon, else it reflects
    if (p->hopR > 0 && p->oHop)
      site = pmcHopSite(p, site, prd);
    if (pmcLoad(p->bandState + site) != viennaps::pmcSi) {
      pmcCount(p, viennaps::pmcOOccupied);
      return;
    }
    if (viennacore::getNextRand(&prd->RNGstate) < p->stickO) {
      // the claim is atomic: of two O on one bare cell the second finds SiO
      if (atomicCAS(p->bandState + site, viennaps::pmcSi, viennaps::pmcSiO) ==
          viennaps::pmcSi) {
        pmcCount(p, viennaps::pmcOBare);
        pmcRecord(p, site);
        prd->rayWeight = 0.f;
        return;
      }
      pmcCount(p, viennaps::pmcOOccupied);
      return;
    }
    pmcCount(p, viennaps::pmcOBare);
    pmcCount(p, viennaps::pmcStickFail);
    return;
  }

  // F: one cascade step on the reaction site, unless that site is SiO
  pmcCount(p, viennaps::pmcFArrivals);
  if (p->hopR > 0 && p->fHop)
    site = pmcHopSite(p, site, prd);
  if (pmcLoad(p->bandState + site) == viennaps::pmcSiO) {
    pmcCount(p, viennaps::pmcFonO);
    return;
  }
  const int rk = min(pmcLoad(p->bandStep + site), p->cascadeTop);
  if (viennacore::getNextRand(&prd->RNGstate) < p->cascadeP[rk]) {
    const int before = atomicAdd(p->bandStep + site, 1);
    if (before >= p->cascadeTop)
      atomicExch(p->bandState + site, viennaps::pmcGone); // SiF4 takes the Si
    else
      atomicCAS(p->bandState + site, viennaps::pmcSi, viennaps::pmcSiF);
    pmcCount(p, site == static_cast<int>(hit) ? viennaps::pmcHopStay
                                              : viennaps::pmcHopMoved);
    pmcRecord(p, site);
    prd->rayWeight = 0.f; // consumed into the bond
    return;
  }
  pmcCount(p, viennaps::pmcStickFail);
}

/// Re-emission of a neutral that did not react: cosine about the cell's
/// estimator normal, restarted one cell out along it (psVoxelPMC::reemitFrom).
/// In 2D the pipeline drops the z component of the 3D cosine sample, which is
/// the CPU's ReflectionDiffuse<T, 2>.
__forceinline__ __device__ void
pmcNeutralReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  if (prd->rayWeight <= 0.f)
    return;
  const auto *p = pmcParams();
  if (prd->numReflections >= static_cast<unsigned int>(p->maxBounce)) {
    pmcCount(p, viennaps::pmcBounceCap);
    prd->rayWeight = 0.f;
    return;
  }
  const float walk = p->walkOut[prd->primID];
  if (walk < 0.f) {
    pmcCount(p, viennaps::pmcNoEscape);
    prd->rayWeight = 0.f;
    return;
  }
  const auto n = viennaray::gpu::getNormal(sbtData, prd->primID);
  viennaray::gpu::diffuseReflection(prd, n); // to the hit point, cosine about n
  for (int d = 0; d < 3; ++d)
    prd->pos[d] += walk * n[d];
  prd->numBoundaryHits = 0; // the side-mirror cap is per segment, as on the CPU
}

//
// --- the ion
//

/// psVoxelPMC::reflectedEnergy: the energy kept after a glancing reflection.
__forceinline__ __device__ float
pmcReflectedEnergy(const viennaps::VoxelPMCParamsGPU *p,
                   viennaray::gpu::PerRayData *prd, float incAngle) {
  const float inflect = p->inflectAngle;
  const float A = 1.f / (1.f + p->n_l * (M_PI_2f / inflect - 1.f));
  const float peak =
      incAngle >= inflect
          ? 1.f - (1.f - A) * (M_PI_2f - incAngle) / (M_PI_2f - inflect)
          : A * powf(incAngle / inflect, p->n_l);
  const float E = prd->energy;
  float out = peak * E + 0.1f * E * viennacore::getNormalDistRand(&prd->RNGstate);
  for (int i = 0; i < 64 && (out < 0.f || out > E); ++i)
    out = peak * E + 0.1f * E * viennacore::getNormalDistRand(&prd->RNGstate);
  return fminf(fmaxf(out, 0.f), E);
}

/// psVoxelPMC::reflectConed in 2D: the specular direction off n, turned IN
/// THE PLANE by the coned polar angle, to either side. The in-plane traceDir
/// is the ion's direction; prd->dir may still carry the z component the 2D
/// source folds away. In 3D the CPU's cone is ViennaRay's, used as it is.
__forceinline__ __device__ void
pmcConedInPlane(viennaray::gpu::PerRayData *prd, const viennacore::Vec3Df &n,
                const float maxCone) {
  for (int d = 0; d < 3; ++d)
    prd->pos[d] += prd->tMin * prd->traceDir[d]; // to the hit point
  const auto &v = prd->traceDir;
  const float dn = v[0] * n[0] + v[1] * n[1];
  float w0 = v[0] - 2.f * dn * n[0], w1 = v[1] - 2.f * dn * n[1];
  const float wl = sqrtf(w0 * w0 + w1 * w1);
  if (wl > 0.f) {
    w0 /= wl;
    w1 /= wl;
  }
  float theta = 0.f;
  if (maxCone > 0.f) {
    for (int i = 0;; ++i) {
      const float u = sqrtf(viennacore::getNextRand(&prd->RNGstate));
      const float sq = sqrtf(fmaxf(1.f - u, 0.f));
      theta = maxCone * sq;
      if (viennacore::getNextRand(&prd->RNGstate) * theta * u <=
          cosf(M_PI_2f * sq) * sinf(theta))
        break;
      if (i > 64) {
        theta = 0.f;
        break;
      }
    }
  }
  float t0 = -w1, t1 = w0;
  if (viennacore::getNextRand(&prd->RNGstate) < 0.5f) {
    t0 = -t0;
    t1 = -t1;
  }
  const float st = sinf(theta), ct = cosf(theta);
  float o0 = st * t0 + ct * w0, o1 = st * t1 + ct * w1;
  const float dp = o0 * n[0] + o1 * n[1]; // keep it in the gas hemisphere
  if (dp <= 0.f) {
    o0 -= 2.f * dp * n[0];
    o1 -= 2.f * dp * n[1];
  }
  const float ol = sqrtf(o0 * o0 + o1 * o1);
  if (ol > 0.f) {
    o0 /= ol;
    o1 /= ol;
  }
  prd->dir = viennacore::Vec3Df{o0, o1, 0.f};
}

__forceinline__ __device__ void pmcIonInit(viennaray::gpu::PerRayData *prd) {
  const auto *p = pmcParams();
  viennaps::gpu::impl::initNormalDistEnergy(prd, p->meanEnergy, p->sigmaEnergy);
}

/// An ion impact on a non-mask cell is an event: the host spreads it to its
/// reaction site, takes the yield of that site's identity and removes cells.
__forceinline__ __device__ void
pmcIonCollision(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  const auto *p = pmcParams();
  const unsigned int hit = prd->primID;
  if (pmcDropped(p, prd, hit))
    return;
  if (!p->normalValid[hit])
    pmcCount(p, viennaps::pmcNoNormalHit);
  if (p->bandState[hit] == viennaps::pmcMask)
    return; // the mask: nothing removed, it may still reflect
  const auto n = viennaray::gpu::getNormal(sbtData, hit);
  const float cosT = __saturatef(-viennacore::DotProduct(prd->traceDir, n));
  pmcCount(p, viennaps::pmcIonImpacts);
  const unsigned int k = atomicAdd(p->eventCount, 1u);
  if (k < p->eventCapacity) {
    p->ionEvents[4 * k + 0] = static_cast<float>(hit);
    p->ionEvents[4 * k + 1] = prd->energy;
    p->ionEvents[4 * k + 2] = cosT;
    p->ionEvents[4 * k + 3] = prd->numReflections > 0 ? 1.f : 0.f;
  } else {
    pmcCount(p, viennaps::pmcOverflow);
  }
}

/// Reflect or stop, as the CPU ion (ChemicalIon::surfaceReflection): steep
/// hits stop, glancing ones reflect with a reduced energy into a cone about
/// the specular direction off the smoothed wall, restarted one cell out.
__forceinline__ __device__ void
pmcIonReflection(const void *sbtData, viennaray::gpu::PerRayData *prd) {
  if (prd->rayWeight <= 0.f)
    return;
  const auto *p = pmcParams();
  const unsigned int hit = prd->primID;
  const auto n = viennaray::gpu::getNormal(sbtData, hit);
  const float cosT = __saturatef(-viennacore::DotProduct(prd->traceDir, n));
  const float incAngle = acosf(cosT);
  float sticking = 1.f;
  if (incAngle > p->thetaRMin)
    sticking = 1.f - __saturatef((incAngle - p->thetaRMin) /
                                 (p->thetaRMax - p->thetaRMin));
  if (sticking >= 1.f ||
      viennacore::getNextRand(&prd->RNGstate) < sticking ||
      prd->numReflections >= static_cast<unsigned int>(p->maxBounce)) {
    prd->rayWeight = 0.f;
    return;
  }
  const float newE = pmcReflectedEnergy(p, prd, incAngle);
  if (newE <= p->minEth) {
    prd->rayWeight = 0.f;
    return;
  }
  prd->energy = newE;
  pmcCount(p, viennaps::pmcIonBounce);
  viennacore::Vec3Df nRef = n;
  {
    const float *f = p->fitNormal + 3 * hit;
    const float l = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
    if (l > 0.5f)
      nRef = viennacore::Vec3Df{f[0], f[1], f[2]};
  }
  const float cone = M_PI_2f - fminf(incAngle, p->minAngle);
  if (p->D == 2)
    pmcConedInPlane(prd, nRef, cone);
  else
    viennaray::gpu::conedCosineReflection(prd, nRef, cone);
  const float walk = p->walkOut[hit];
  if (walk < 0.f) {
    pmcCount(p, viennaps::pmcNoEscape);
    prd->rayWeight = 0.f;
    return;
  }
  for (int d = 0; d < 3; ++d)
    prd->pos[d] += walk * n[d];
  prd->numBoundaryHits = 0; // the side-mirror cap is per segment, as on the CPU
}
