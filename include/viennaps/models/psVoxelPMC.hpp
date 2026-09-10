#pragma once

#include <csVoxelInteraction.hpp>
#include <materials/psMaterialMap.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <cstdint>
#include <random>
#include <vector>

namespace viennaps {

using namespace viennacore;

/// A BINARY-cell particle Monte Carlo, beside the filling-fraction arm.
///
/// DEFAULTS, settled on the spontaneous (fluorine-only) benchmark:
///
///   simpleFlux_   a particle hits a cell and either removes it or reflects.
///                 No coverage state of any kind -- the arm is per impact.
///   planeAccept_  a hit counts only if the ray crosses the RECONSTRUCTED
///                 surface (the least-squares plane through the surface cell
///                 centres) within planeWindow_ = 1 cell of where it entered.
///                 Against cell faces a grazing ray clips a step the real
///                 shape does not have, and each spurious hit is another
///                 chance to react: measured 1.16 hits per particle against
///                 1.00 with this test.
///   minFitPts_    below six surface cells the fit is not trusted and the hit
///                 is taken, so a cell with no meaningful local plane still
///                 etches.
///   prune_        whole-cell removal strands islands. The caller clears them
///                 ONCE after the run (pruneUnsupported); doing it per step
///                 runs the etch ahead of itself.
///
/// offPlaneTest_ is off: it was aimed at the same islands and costs the
/// profile shape. Pruning is the better answer.
///
/// The two arms share transport: this class drives viennacs::VoxelInteraction
/// -- the same traversal, the same BVH, the same source law -- so a comparison
/// between them isolates the REPRESENTATION and nothing else.
///
/// What differs is everything downstream of a hit.
///
///   filling fraction        f in [0,1] per cell; rays deposit a flux field;
///                           coverages solve from that field; a velocity moves
///                           the fill; overflow redistributes.
///
///   this class             f in {0,1}; each ray fires ONE event at the cell it
///                           hits; the cell's own state carries the chemistry;
///                           a cell either survives or turns to gas.
///
/// So there is no flux field, no coverage solve, no velocity, and no advance
/// -- the whole redistribution and anchoring machinery exists to move a
/// fractional interface and has nothing to move here.
///
/// THE STATE IS THE CELL. A surface cell is Bare, Fluorinated or Oxidised, and
/// theta is never stored: it is the fraction of surface cells in each state,
/// recovered by counting when a diagnostic asks for it.
///
/// TWO CONVERSIONS carry the continuum model onto discrete cells:
///
///   a yield is in ATOMS, a cell holds rho*dx^3 of them, so an ion removes
///       n = Y / (rho*dx^3)   cells,
///   an expected COUNT which may exceed one (remove floor(n), plus one more
///   with the remainder). Keeping n < 1 wants dx > (Ymax/rho)^(1/3).
///
///   a thermal step is driven by TIME, not by arrivals: an F cell fires at
///   4*k_sigma*dx^2 and reverts to Bare, and that same firing removes the cell
///   with probability 1/(4*rho*dx^3). Tying it to F arrivals instead gives the
///   right rate only at steady state, where arrival balances consumption.
///
/// PROTOTYPE LIMIT -- the ray loop is SERIAL. Events do not commute: if one ray
/// removes a cell, a later ray should fly through the hole, so a parallel loop
/// needs batched events applied in a fixed order, and the batch size then has
/// to be validated. Correctness first; that is a separate piece of work.
template <class NumericType, int D> class VoxelPMC {
public:
  enum State : std::uint8_t { Bare = 0, Fluorinated = 1, Oxidised = 2 };

  /// SF6/O2, the numbers of sf6o2.mechanism.json. Yields are
  /// Y = A*(sqrt(E)-sqrt(Eth))*f(theta), in ATOMS per ion.
  struct Parameters {
    NumericType fluxF = 1800, fluxO = 100, fluxIon = 12;
    NumericType stickF = 0.7, stickO = 1.0;
    NumericType kSigma = 75, betaSigma = 0.04;
    NumericType rho = 50.2;   ///< silicon atoms per nm^3
    NumericType meanEnergy = 100, sigmaEnergy = 10;
    NumericType cosinePowerNeutral = 1, cosinePowerIon = 500;
    NumericType A_sp = 0.0337, Eth_sp = 20, B_sp = 9.3; ///< sputter Si
    NumericType A_ie = 7.0, Eth_ie = 15;                ///< ion-enhanced Si
    NumericType A_p = 3.0, Eth_p = 10;                  ///< clear O
    /// Areal density of adsorption sites. The mechanism quotes fluxes in
    /// 1e15/cm^2/s = 10/nm^2/s and rho in 1e22/cm^3 = 10/nm^3, so the same
    /// number serves as an areal flux in ER and a per-site rate in the
    /// coverage balance only if sigma0 = 10/nm^2 -- which sits between the
    /// Si(100) surface density 6.8 and rho^(2/3) = 13.6.
    NumericType sigma0 = 10.0;
    /// Ion reflection, the ionSource block of the mechanism. Below
    /// thetaRMin the ion sticks; between thetaRMin and thetaRMax it reflects
    /// with a coned-cosine direction and a reduced energy. Without this the
    /// ion-enhanced channel -- 84 % of the removal -- loses every grazing
    /// ion instead of letting it carry its yield deeper.
    NumericType thetaRMin = 70, thetaRMax = 90;   ///< degrees
    NumericType minAngle = 80, inflectAngle = 89; ///< degrees
    NumericType n_l = 10;
  };

private:
  const viennacs::LatticeMap<NumericType, D> *lattice_;
  std::vector<NumericType> *fill_; ///< kept binary: 0 or 1
  std::vector<int> *material_;
  std::vector<std::uint8_t> state_;
  viennacs::VoxelInteraction<NumericType, D> interaction_;
  Parameters p_;
  std::mt19937_64 rng_{1};
  size_t removedCells_ = 0;
  /// Optional sub-cell accounting: atoms owed by each COLUMN, carried across
  /// events instead of being rounded to a whole cell on the spot. The surface
  /// stays binary -- transport still sees a staircase -- only the moment a
  /// cell flips changes. It removes the rounding variance, not the variance
  /// in the dose itself.
  /// DEFAULTS. Measured against the level set on a W=40 nm masked trench,
  /// 3 seeds: floor depth -24.8% with none of these on, -5.2% with all three,
  /// and the roughness sigma falls 2.38 -> 0.93 nm. Re-emission is the one
  /// that matters for accuracy; the other two buy smoothness.
  bool fractional_ = true;
  std::vector<NumericType> credit_;
  /// Super-particles on the ION channel: trace `ionWeight_` times as many
  /// ions, each carrying 1/ionWeight_ of the yield. The mean is unchanged and
  /// the removal noise falls as 1/sqrt(ionWeight_) -- but only with
  /// fractional accounting, since otherwise the variance sits in the integer
  /// count of cells removed and no amount of slicing touches it. Ions are a
  /// few per cent of the particle budget, so this is nearly free.
  NumericType ionWeight_ = 4;
  /// Draw the fluorine an ion consumes from ITS OWN neighbourhood. The global
  /// walk lets an ion on the trench floor strip a sidewall 30 nm away, which
  /// matters because a staircase carries far more surface cells than a smooth
  /// surface and most of them are side faces a collimated ion never strikes.
  /// An ion's cascade is LOCAL. Clearing adsorbate by a global walk lets an
  /// ion on the trench floor strip oxygen off a sidewall 30 nm away, and the
  /// wall -- which sees almost no ions of its own, so R6 cannot fire there and
  /// R7 desorbs at only beta = 0.04 /s -- should instead saturate with oxygen.
  bool localClear_ = true;
  /// Bounce a neutral that fails to stick instead of discarding it. The
  /// continuum arms do this (csVoxelFlux re-emits about the local normal),
  /// and a neutral needs several bounces to reach a trench floor, so without
  /// it the feature is starved. Ions stay single-hit: they are collimated.
  bool reemit_ = true;
  /// Search distance for the normal fit, in cells. MCFPM uses 4*dx.
  ///
  /// Set from the sputtering-only benchmark -- ions on bare silicon, no
  /// fluorine, no oxygen, no ion-enhanced channel -- where the incidence
  /// angle is the ONLY ingredient and f_sp = (1 + 9.3 sin^2 t) cos t peaks
  /// about 4x near 53 deg. On a W=40 nm trench etched 8 nm the profile
  /// against the level set is:
  ///
  ///   radius 2      floor sits high, the central mound rides above
  ///   radius 3      tracks the contour across the floor and both corners
  ///   radius 4      as good at the corners, looser in the middle
  ///   radius 8, 12  corner troughs to -18/-19 nm against the level set's -12
  ///
  /// A WIDE stencil averages the corner's real curvature away, so cells at
  /// the wall foot report a tilt closer to the wall than they have, pushing
  /// theta into the range where f_sp is amplified. The answer is flat over
  /// 2--4 and degrades sharply beyond, so the exact value does not matter
  /// much inside that band; 3 is also where the estimator itself is most
  /// accurate against voxelised planes (2.5 deg, against 3.1 at R = 2).
  ///
  /// An earlier value of 12 came from the ion-driven trench measured with
  /// the coverage model, the grid-walk transport and an unseeded accumulator
  /// all in play, and does not survive any of those being fixed.
  int fitRadius_ = 3;
  /// Separate stencil for the SPECULAR direction only, defaulting to the
  /// same value.
  ///
  /// MEASURED TO DO NOTHING on a masked trench. Crossing yield radius with
  /// specular radius, {4,12} x {4,12}, 3 seeds, with and without the
  /// curvature cap, the main effects are
  ///
  ///   volume        yield 4->12  -14.0 pts    specular 4->12  -1.4 pts
  ///   microtrench   yield 4->12  +3.29 nm     specular 4->12  -0.25 nm
  ///
  /// and with the cap off the specular effect falls to -0.3 pts and
  /// -0.01 nm. The angle that matters is the INCIDENCE angle feeding the
  /// yields, not the direction a reflected ion leaves in. Kept separate
  /// because a geometry with long specular paths may yet need it, but do not
  /// expect tuning it to move a trench.
  int reflectRadius_ = 3;
  bool freeze_ = false;   ///< adsorb and desorb, but never remove material
  bool sideReflect_ = true;  ///< mirror rays at the lateral walls
  int areaNorm_ = 0;         ///< 0 none, 1 face count, 2 true area 1/max|n|
  bool fBalance_ = false;    ///< spontaneous etch from the F balance
  bool handDown_ = false;    ///< a removed cell passes its state downward
  /// SITE OCCUPANCY. A cell stands for nu = sigma0*dx^(D-1) adsorption
  /// sites, and a single [F]/[Si] flag can only say "all of them" or "none".
  /// A cell holding 9 of 10 fluorines then still accepts arrivals at full
  /// probability, and one holding 1 rejects them all. Counting the occupied
  /// sites instead makes the acceptance (nu-nF-nO)/nu, which is what the
  /// site balance actually asks for. The GEOMETRY stays binary: a cell is
  /// solid or gas and one event still removes a whole cell.
  bool siteCounts_ = false;
  std::vector<std::uint8_t> nF_, nO_;
  /// FLUX-DRIVEN spontaneous etch: no coverage state at all.
  ///
  /// At steady state theta is slaved to the local flux,
  ///     theta = G*s0 / (G*s0 + 4*k_sigma),   G = arrivals per SITE per second
  /// and the removal is k_sigma*theta per site. Counting the arrivals each
  /// cell actually receives and applying that closed form is what the level
  /// set does, and it removes the coverage dynamics from the comparison
  /// entirely: what is left is purely a question of transport.
  bool fluxEtch_ = false;
  std::vector<std::uint32_t> arrF_;
  /// SIMPLE FLUX ETCH. One particle, one cell, one decision: an F that hits
  /// a silicon cell either removes it or reflects. No state, no coverage, no
  /// conversion between species. The probability follows from the steady
  /// state of the same two reactions:
  ///
  ///   theta = fluxF*s0 / (fluxF*s0 + 4*k_sigma)      (= 0.808)
  ///   atoms removed per arriving F = s0*(1-theta)/4  (= 0.0336)
  ///   p(remove the cell hit) = that / (rho*dx^D)     (= 6.7e-4 at dx = 1)
  bool simpleFlux_ = true;   ///< DEFAULT: coverage-free, per-hit removal
  NumericType simpleP_ = -1;   ///< >=0 overrides the derived p
  /// Distance gate on a RE-EMITTED ray, in cells: it may not interact
  /// until it has travelled this far from where it left the surface.
  /// The filling-fraction arm carries the same gate.
  NumericType armAfter_ = 0;
  bool planeAccept_ = true;       ///< DEFAULT: test hits against the fitted plane
  NumericType planeWindow_ = 1.0; ///< cells from entry to the crossing
  int minFitPts_ = 6;             ///< below this the fit is not trusted
  bool prune_ = true;             ///< DEFAULT: caller prunes after the run
  /// Accept a hit when the fitted plane misses the cell itself. Aimed at
  /// stranded cells, but it also lets grazing floor hits back in and
  /// costs the profile shape (lateral 27.5 -> 25.5 nm on a W=40 trench),
  /// so island pruning is the better answer and this stays off.
  bool offPlaneTest_ = false;
  /// CONFORMAL DEPOSITION. The mirror of the simple flux etch, and the same
  /// single decision: a precursor that hits a solid cell either STICKS -- and
  /// is consumed there -- or reflects diffusively and flies on. A stuck
  /// molecule hands `depAtoms_` atoms to the gas cell the ray last crossed;
  /// that cell turns solid once a whole cell's worth has arrived, so growth
  /// is quantised in exactly the way removal is.
  ///
  /// depP_ is the NET sticking probability s0*(1-theta) taken from the
  /// mechanism's own steady state, so the blanket growth rate is
  /// flux*sigma0*depP_/rho. Lowering it is what makes the film conformal:
  /// a molecule that does not stick is still in the trench and gets another
  /// chance further down.
  bool deposit_ = false;
  NumericType depP_ = 1;
  NumericType depAtoms_ = 1;      ///< solid atoms carried by one molecule
  int filmMaterial_ = -1;         ///< <0: adopt the cell grown on
  std::vector<NumericType> depCredit_;  ///< atoms banked, per cell
  /// Re-emissions a neutral may make before it is discarded. With a
  /// small per-hit reaction probability a particle needs many bounces
  /// to react at all, and the ones that would reach a shadowed region
  /// are the first to be cut off by a low budget.
  int kMaxBounce = 24;
  static constexpr int kMaxSideBounce = 8;
  ///< real reflections per particle
  /// Pass-throughs are NOT reflections: a ray declining a ledge keeps its
  /// direction and advances only 1.5 dx, so crossing a feature costs O(1/dx)
  /// of them. Sharing one fixed budget with the reflections killed rays
  /// before they crossed the trench on a fine grid -- a dx-dependent loss of
  /// flux. Scale the allowance with the lattice instead.
  int passBudget() const {
    int span = 0;
    for (int d = 0; d < D; ++d)
      span += lattice_->dims()[d];
    return 4 * span;
  }
  /// max of sum|n_i| * cos(theta) over orientations: (1+sqrt2)/2 in 2D,
  /// ~1.366 in 3D (maximise (sqrt2*sqrt(1-c^2)+c)*c). Acceptance is divided
  /// by it and the emitted particle count multiplied by it, so the angular
  /// response is exact and the absolute flux unchanged.
  static constexpr NumericType kFluxBoost = D == 2 ? NumericType(1.2071068)
                                                   : NumericType(1.3660254);

public:
  // event counters, to check the balance against the continuum rates
  size_t nAdsF = 0, nAdsO = 0, nClearF = 0, nClearO = 0, nThermF = 0;
  size_t nIons = 0, nIonsOnF = 0, nSurf = 0;
  size_t nBounce = 0, nBounceFail = 0, nBounceCap = 0, nEscape = 0;
  size_t nLaunch = 0;   ///< particles emitted, to get hits/particle
  size_t nHitN = 0;     ///< neutral surface hits actually processed
  size_t nRem0 = 0, nRemB = 0;  ///< removals on the FIRST hit vs after a reflection
  size_t nPruned = 0;           ///< cells removed as unsupported islands
  size_t nAcc = 0, nAccClamped = 0; double sumAccRaw = 0;
  double sumCosIon = 0; size_t nCosIon = 0;   ///< angle ions see
  size_t nIonBounce = 0;                      ///< ion reflections
  size_t nSideMirror = 0;                     ///< rays mirrored at a side wall
  /// Neighbours reset to Bare because a removal uncovered them. Per
  /// removed cell this is the rate at which fresh, uncovered silicon is
  /// created, and it is what the adsorption has to keep up with. A 3D
  /// removal can uncover more neighbours than a 2D one.
  size_t nBareReset = 0;
  /// The fractional ledger. atomsOwed is everything the yields asked
  /// for; removedCells*atomsPerCell is what was actually taken; the
  /// difference must be the credit still outstanding, or atoms are
  /// being lost. nRemoveFail counts the times a column had a whole
  /// cell's worth owed but nothing could be removed for it.
  double atomsOwed = 0;
  size_t nRemoveFail = 0;
  size_t remIE = 0, remSp = 0, remTh = 0;     ///< cells, by channel
  size_t nDepStick = 0;   ///< precursors consumed by the surface
  size_t nDepFail = 0;    ///< stuck, but with nowhere to put the atoms
  size_t depositedCells_ = 0;
  /// where the ions land and where they bite, binned by lateral cell index
  std::vector<size_t> histHit, histOnF, histRem, histTh;
  /// ion response vs local surface angle: 10-degree bins of theta
  std::array<size_t, 9> angHit{}, angRem{};
  /// ion-enhanced yield by incidence angle, split by WHERE it landed:
  /// a wall cell (exposed laterally, not from above) or a floor cell
  std::array<size_t, 9> wallHit{}, floorHit{};
  /// REFLECTED ions only (bounce > 0): where they deposit ion-enhanced yield,
  /// binned laterally, and the angle they arrive at. This is the microtrench
  /// channel -- an ion grazing the wall reflects down onto the foot of it and
  /// lands near 60 deg, where f_sp peaks at ~4x and f_ion is still unity.
  std::vector<size_t> reflRem, directRem;
  std::array<size_t, 9> reflAng{};
  std::array<size_t, 12> bounceHist{};   ///< reflections per ion
  double reflY = 0, directY = 0;
  std::array<double, 9> wallY{}, floorY{};
  std::array<double, 9> angArea{};

private:

  NumericType delta() const { return lattice_->gridDelta(); }

  /// Reflective LATERAL boundaries, as the level-set arm uses.
  ///
  /// A ray that leaves the side of the domain was simply dropped. On a
  /// feature sitting in the middle of a wide domain that is harmless, but the
  /// dose is then short in a border band as wide as the source is high, and a
  /// 3D box has that band on four sides where 2D has it on two -- which is
  /// most of why every 3D result here came out worse than its 2D counterpart.
  /// Measured on a blanket, whose answer must be the analytic rate: 2D -1.0 %
  /// at extent 40 against 3D -17.7 %, converging to -0.5 % and -8.7 % only by
  /// extent 160. The 3D hole harness runs at extent 40.
  ///
  /// Mirroring the ray at the side wall is what a reflective boundary means
  /// and costs one plane intersection. Rays leaving through the TOP still
  /// escape: that is a real loss, not a boundary artefact.
  bool mirrorAtSide(std::array<NumericType, D> &o,
                    std::array<NumericType, D> &dir) const {
    const auto &mn = lattice_->minCorner();
    const auto &dims = lattice_->dims();
    constexpr NumericType kEps = 1e-12;
    NumericType tSide = std::numeric_limits<NumericType>::max();
    int axis = -1;
    for (int d = 0; d < D - 1; ++d) {
      const NumericType lo = mn[d];
      const NumericType hi = mn[d] + delta() * static_cast<NumericType>(dims[d]);
      NumericType t = -1;
      if (dir[d] > kEps) t = (hi - o[d]) / dir[d];
      else if (dir[d] < -kEps) t = (lo - o[d]) / dir[d];
      if (t > 0 && t < tSide) { tSide = t; axis = d; }
    }
    if (axis < 0)
      return false;
    // If it would leave through the top first, it has genuinely escaped.
    if (dir[D - 1] > kEps) {
      const NumericType hiZ =
          mn[D - 1] + delta() * static_cast<NumericType>(dims[D - 1]);
      if ((hiZ - o[D - 1]) / dir[D - 1] < tSide)
        return false;
    }
    for (int d = 0; d < D; ++d)
      o[d] += dir[d] * tSide;
    dir[axis] = -dir[axis];
    o[axis] += dir[axis] * delta() * NumericType(1e-3);  // nudge inside
    return true;
  }

  /// firstHit, but a ray that exits the side is mirrored back in rather than
  /// lost. Bounded so a ray trapped in a corner cannot loop forever.
  viennacs::VoxelHit<NumericType, D>
  traceReflective(std::array<NumericType, D> &o,
                  std::array<NumericType, D> &dir,
                  NumericType armAfter = 0) {
    auto h = interaction_.firstHit(o, dir, rng_, armAfter);
    if (!sideReflect_)
      return h;
    for (int k = 0; !h.hit() && k < kMaxSideBounce; ++k) {
      if (!mirrorAtSide(o, dir))
        break;
      ++nSideMirror;
      h = interaction_.firstHit(o, dir, rng_, armAfter);
    }
    return h;
  }
  NumericType uni() {
    return std::uniform_real_distribution<NumericType>(0, 1)(rng_);
  }

  bool solid(int id) const { return id >= 0 && (*fill_)[id] >= NumericType(0.5); }

  bool isMask(int id) const {
    return id >= 0 && (*material_)[id] == static_cast<int>(Material::Mask);
  }

  /// a solid cell with at least one gas face-neighbour
  /// Exposed solid, MASK INCLUDED. A mask cell bears no chemistry, so
  /// isSurface() excludes it -- but it is still a perfectly good reflector,
  /// and a ray that lands on one must bounce, not be discarded. In a hole the
  /// mask sidewall is several times the opening area, so dropping those rays
  /// starves the floor of everything.
  bool isExposed(const std::array<int, D> &idx) const {
    const int id = lattice_->cellId(idx);
    if (!solid(id))
      return false;
    for (int d = 0; d < D; ++d)
      for (int s = -1; s <= 1; s += 2) {
        auto n = idx;
        n[d] += s;
        const int nid = lattice_->cellId(n);
        if (nid >= 0 && !solid(nid))
          return true;
      }
    return false;
  }

  bool isSurface(const std::array<int, D> &idx) const {
    const int id = lattice_->cellId(idx);
    if (!solid(id) || isMask(id))
      return false;
    for (int d = 0; d < D; ++d)
      for (int s = -1; s <= 1; s += 2) {
        auto n = idx;
        n[d] += s;
        const int nid = lattice_->cellId(n);
        // A neighbour off the lattice CONTINUES the field rather than being
        // gas. Treating it as gas made every boundary cell -- both side
        // columns and the whole bottom row -- count as surface: buried,
        // permanently bare, and diluting every coverage this class reports.
        if (nid >= 0 && !solid(nid))
          return true;
      }
    return false;
  }

  /// Restart a ray just OUTSIDE the interface it came from, pointing in a
  /// cosine-distributed direction about the local normal. Stepping past the
  /// last cell holding material is what stops a ray re-interacting with the
  /// same surface it just left.
  /// sum|n_i| for a cell: the true area it carries, in units of dx^(D-1).
  ///
  /// Count it on a plane crossing the grid. Horizontal: one exposed cell per
  /// dx of surface, sum|n| = 1. At 45 deg: one cell per sqrt(2) dx, sum|n| =
  /// sqrt(2). Vertical: one cell per dx of wall, sum|n| = 1. So a cell holds
  /// nu*sum|n_i| SITES, and every per-site rate carries that factor.
  /// Radius, in cells, of the DAMAGE cascade -- the volume an ion disturbs,
  /// which is what sets how far material removal can reach.
  ///
  /// This is NOT cascadeRadius(2Y). That one is the surface patch over which
  /// the ion consumes ADSORBATE, 2Y/sigma0 = 8.58 nm^2, and using it for
  /// removal lets an ion on the trench floor take a cell 4.29 nm away --
  /// measured as a 2.3 nm flare under the mask where the level set undercuts
  /// 0.09 nm. The cascade that actually displaces atoms is the sphere holding
  /// Y atoms, radius (3Y/4*pi*rho)^(1/3) ~ 1.2 nm at 100 eV, which is also
  /// the right order for a 100 eV Ar ion in silicon.
  NumericType damageRadius(NumericType Y) const {
    const NumericType r = std::cbrt(3 * Y / (4 * NumericType(M_PI) * p_.rho));
    return r / delta();
  }

  /// Radius, in cells, of the surface patch holding `sites` adsorbate sites.
  /// The patch has area sites/sigma0. In 3D that is a disc, R = sqrt(A/pi);
  /// in 2D it is a length per unit depth, so the half-length is A/2. At
  /// Y_ie = 42.9 this is 1.65 cells in 3D but 4.29 in 2D -- using the 3D
  /// form in 2D searches a patch 2.6x too small.
  NumericType cascadeRadius(NumericType sites) const {
    const NumericType area = sites / p_.sigma0;      // nm^(D-1)
    if constexpr (D == 3)
      return std::sqrt(area / NumericType(M_PI)) / delta();
    else
      return area / (2 * delta());
  }

  /// sites a cell at `idx` holds; the empty index means "use the mean"
  /// sites a cell stands for, before any area weighting
  NumericType nuScalar() const {
    return p_.sigma0 * std::pow(delta(), D - 1);
  }
  /// fraction of a cell's sites that are free / fluorinated
  NumericType freeFrac(int id) const {
    if (!siteCounts_)
      return state_[id] == Bare ? NumericType(1) : NumericType(0);
    const NumericType nu = nuScalar();
    const NumericType occ = static_cast<NumericType>(nF_[id] + nO_[id]);
    return occ >= nu ? NumericType(0) : (nu - occ) / nu;
  }
  NumericType fFrac(int id) const {
    if (!siteCounts_)
      return state_[id] == Fluorinated ? NumericType(1) : NumericType(0);
    return static_cast<NumericType>(nF_[id]) / nuScalar();
  }
  /// keep the visualised state in step with the counts
  void syncState(int id) {
    if (!siteCounts_) return;
    state_[id] = nF_[id] >= nO_[id] ? (nF_[id] ? Fluorinated : Bare)
                                    : Oxidised;
  }
  NumericType nuOf(const std::array<int, D> &idx) const {
    const NumericType nu = p_.sigma0 * std::pow(delta(), D - 1);
    bool zero = true;
    for (int d = 0; d < D; ++d)
      if (idx[d] != 0) zero = false;
    return zero ? nu * NumericType(1.2) : nu * areaFactor(idx);
  }

  /// True interface area a cell carries, in units of dx^(D-1).
  ///
  /// This is what csVoxelAdvance::interfaceArea computes and csVoxelFlux
  /// divides its deposits by: |grad f| is the interfacial area per unit
  /// VOLUME, so the area is |grad f| * dx^D. It is NOT the staircase area
  /// sum|n_i| * dx^(D-1) -- that counts exposed cell FACES, which over-counts
  /// wherever a cell shows more than one, at most 2 in 2D but 3 in 3D. Using
  /// the staircase form is what left 2D within a few percent while 3D sat at
  /// -28 %.
  /// MCFPM applies NO per-cell area normalisation: "when a gas particle
  /// intersects with a surface, the probability of reaction is, by
  /// definition, unity" (Huard 2.5.5). The staircase-vs-true-area problem is
  /// solved GEOMETRICALLY there, by reconstructing a continuous surface from
  /// the cell centres, not by weighting. The only areal bridge we need is
  /// nu = sigma0*dx^(D-1), because our mechanism is written per unit area
  /// while theirs is written per impact.
  ///
  /// areaNorm_ switches that off: a cell showing k exposed faces intercepts
  /// the ray field through all k of them while holding one cell of material,
  /// so it is dosed k times too heavily for what it can give up. A flat
  /// axis-aligned cell has k = 1 and is unaffected -- which is why a blanket
  /// comes out right and curved or cornered geometry does not.
  NumericType areaFactor(const std::array<int, D> &idx) const {
    if (areaNorm_ == 0)
      return NumericType(1);
    if (areaNorm_ == 1) {                     // exposed-face count, 1..2D
      int faces = 0;
      for (int d = 0; d < D; ++d)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          auto nb = idx;
          nb[d] += sgn;
          const int nid = lattice_->cellId(nb);
          if (nid < 0 || !solid(nid))
            ++faces;
        }
      return faces > 0 ? static_cast<NumericType>(faces) : NumericType(1);
    }
    // TRUE interface area a cell carries. For a plane of unit normal n
    // crossing the grid there is one exposed cell per dx^(D-1)/max|n_i| of
    // surface, so the area factor is 1/max|n_i|, between 1 (axis-aligned)
    // and sqrt(D) (diagonal). A staircase cell on a 45 deg slope shows TWO
    // faces but carries only sqrt(2) of area, so counting faces over-states
    // it by 2/sqrt(2) exactly where the surface is sloped.
    const auto n = interaction_.normalAt(idx);
    NumericType mx = 0;
    for (int d = 0; d < D; ++d)
      mx = std::max(mx, std::abs(n[d]));
    return mx > NumericType(0.1) ? NumericType(1) / mx : NumericType(1);
  }



  /// Does this hit count, or does the ray pass on?
  ///
  /// A staircase cell presents a FACE, but the surface it stands for has an
  /// orientation, and the two collect flux differently. A smooth vertical
  /// wall is nearly invisible to a downward-travelling particle; the
  /// staircase standing in for it is a run of horizontal ledges that catch
  /// that flux head-on. Left uncorrected the wall over-collects in exact
  /// proportion to its excess projected area, over-fluorinates, and etches
  /// sideways -- measured theta_F 0.21 on the wall against the 0.085 the
  /// level set's lateral etch implies.
  ///
  /// So accept the hit with |d.n_true| / |d.n_face|: unity on a facet that
  /// already faces the flux correctly, near zero on a ledge of a wall.
  /// The fitted surface supplies the NORMAL only.
  ///
  /// Three attempts to also take the IMPACT POINT from it all failed, and the
  /// measurements rule out the explanation I kept reaching for. The plane
  /// sits inside the material -- measured by voxelising planes at known tilts:
  /// 2D 0.45 cells axis-aligned, 0.38 at 45 deg; 3D 0.44 / 0.29 / 0.25. But
  /// offsetting the plane out to the exposed face by exactly that amount, and
  /// never dropping a particle, changed 2D not at all (+6.3 % either way) and
  /// made 3D worse (-13.9 % -> -23.9 %). So the offset was NOT what made the
  /// earlier attempts fail, and reattributing the impact is not the remedy
  /// for the sidewall over-collection. Left as the normal source only.
  /// Does the ray really meet the SURFACE here, or only a cell face?
  ///
  /// A staircase presents cell faces where the shape it represents has none.
  /// A grazing ray leaving a locally flat region should fly over the steps
  /// and escape; against cell boxes it clips one and interacts. Testing the
  /// ray against the reconstructed plane instead -- the same least-squares
  /// surface through the cell centres that gives the incidence angle -- lets
  /// it pass, while a real wall still stops it.
  ///
  /// Accept when the ray crosses that plane within `planeWindow_` cells of
  /// where it entered the cell; otherwise it passes through to the next.
  bool resolveImpact(viennacs::VoxelHit<NumericType, D> &h,
                     const std::array<NumericType, D> &dir) {
    ++nAcc;
    if (!planeAccept_)
      return true;
    viennacore::Vec3D<NumericType> n;
    std::array<NumericType, D> c{};
    if (!interaction_.fitPlaneAt(h.index, n, c))
      return true;                        // no plane here: keep the hit
    // A cell with too few neighbours has no meaningful local plane -- an
    // isolated or protruding one is exactly that. Rejecting its hits would
    // make it invisible to every ray and it would survive for ever, which is
    // what leaves stranded cells sitting in the gas.
    if (interaction_.lastFitPoints() < minFitPts_)
      return true;
    // Does the plane describe THIS cell? A stranded or protruding cell sitting
    // near the main surface still gathers enough neighbours to fit a plane --
    // just not one that passes anywhere near itself. If its own centre lies
    // further than a cell from that plane, the reconstruction does not
    // represent it and the hit must count, or the cell is never etched.
    if (offPlaneTest_) {
      NumericType off = 0;
      for (int d = 0; d < D; ++d)
        off += (static_cast<NumericType>(h.index[d]) - c[d]) * n[d];
      if (std::abs(off) > NumericType(1))
        return true;
    }
    const auto &mn = lattice_->minCorner();
    NumericType s = 0, dn = 0;
    for (int d = 0; d < D; ++d) {
      const NumericType pc = (h.point[d] - mn[d]) / delta() - NumericType(0.5);
      s += (pc - c[d]) * n[d];
      dn += dir[d] * n[d];
    }
    if (dn >= -NumericType(1e-6))
      return false;                       // travelling away from the surface
    const NumericType t = -s / dn;        // cells to the plane crossing
    if (t < -NumericType(0.5) || t > planeWindow_) {
      ++nAccClamped;
      return false;                       // the surface is elsewhere
    }
    return true;
  }







  /// Push the ray just past the cell it declined, so it can carry on.
  void passThrough(const viennacs::VoxelHit<NumericType, D> &h,
                   const std::array<NumericType, D> &dir,
                   std::array<NumericType, D> &origin) {
    for (int d = 0; d < D; ++d)
      origin[d] = h.point[d] + dir[d] * delta() * NumericType(1.5);
  }

  /// Energy an ion keeps on reflecting, as psIonModelUtil::updateEnergy:
  /// a peak fraction set by the incidence angle, smeared by 10 %.
  NumericType reflectedEnergy(NumericType E, NumericType incAngle) {
    const NumericType inflect = p_.inflectAngle * NumericType(M_PI) / 180;
    const NumericType A = 1 / (1 + p_.n_l * (NumericType(M_PI_2) / inflect - 1));
    const NumericType peak =
        incAngle >= inflect
            ? 1 - (1 - A) * (NumericType(M_PI_2) - incAngle) /
                      (NumericType(M_PI_2) - inflect)
            : A * std::pow(incAngle / inflect, p_.n_l);
    std::normal_distribution<NumericType> dist(peak * E, NumericType(0.1) * E);
    NumericType out = dist(rng_);
    for (int i = 0; i < 64 && (out < 0 || out > E); ++i)
      out = dist(rng_);
    return std::min(std::max(out, NumericType(0)), E);
  }

  /// Coned cosine about the specular direction, as viennaray's
  /// ReflectionConedCosine -- on this class's own RNG.
  std::array<NumericType, D> reflectConed(const std::array<NumericType, D> &dir,
                                          const std::array<NumericType, D> &n,
                                          NumericType maxCone) {
    // specular
    NumericType dn = 0;
    for (int d = 0; d < D; ++d) dn += dir[d] * n[d];
    std::array<NumericType, D> w{};
    for (int d = 0; d < D; ++d) w[d] = dir[d] - 2 * dn * n[d];
    NumericType wl = 0;
    for (int d = 0; d < D; ++d) wl += w[d] * w[d];
    wl = wl > 0 ? std::sqrt(wl) : NumericType(1);
    for (int d = 0; d < D; ++d) w[d] /= wl;
    if (maxCone <= 0)
      return w;
    // polar angle by the same accept-reject
    double theta;
    for (int i = 0;; ++i) {
      const double u = std::sqrt((double)uni());
      const double sq = std::sqrt(std::max(1.0 - u, 0.0));
      theta = maxCone * sq;
      if ((double)uni() * theta * u <= std::cos(M_PI_2 * sq) * std::sin(theta))
        break;
      if (i > 64) { theta = 0; break; }
    }
    // rotate w by theta, in the plane (2D) or about a random azimuth (3D)
    std::array<NumericType, D> t{};
    if constexpr (D == 2) {
      t[0] = -w[1]; t[1] = w[0];
      if (uni() < NumericType(0.5)) { t[0] = -t[0]; t[1] = -t[1]; }
    } else {
      std::array<NumericType, D> a{1, 0, 0};
      if (std::abs(w[0]) > NumericType(0.9)) a = {0, 1, 0};
      t = {w[1] * a[2] - w[2] * a[1], w[2] * a[0] - w[0] * a[2],
           w[0] * a[1] - w[1] * a[0]};
      NumericType tl = 0;
      for (int d = 0; d < D; ++d) tl += t[d] * t[d];
      tl = tl > 0 ? std::sqrt(tl) : NumericType(1);
      for (int d = 0; d < D; ++d) t[d] /= tl;
      std::array<NumericType, D> b{w[1] * t[2] - w[2] * t[1],
                                   w[2] * t[0] - w[0] * t[2],
                                   w[0] * t[1] - w[1] * t[0]};
      const NumericType phi = 2 * NumericType(M_PI) * uni();
      for (int d = 0; d < D; ++d)
        t[d] = std::cos(phi) * t[d] + std::sin(phi) * b[d];
    }
    std::array<NumericType, D> out{};
    const NumericType st = std::sin(theta), ct = std::cos(theta);
    for (int d = 0; d < D; ++d) out[d] = st * t[d] + ct * w[d];
    NumericType dp = 0;                       // keep it in the gas hemisphere
    for (int d = 0; d < D; ++d) dp += out[d] * n[d];
    if (dp <= 0)
      for (int d = 0; d < D; ++d) out[d] -= 2 * dp * n[d];
    NumericType ol = 0;
    for (int d = 0; d < D; ++d) ol += out[d] * out[d];
    ol = ol > 0 ? std::sqrt(ol) : NumericType(1);
    for (int d = 0; d < D; ++d) out[d] /= ol;
    return out;
  }

  /// false when the ray could not be placed in gas -- walking out along the
  /// steepest axis alone does not always escape (a face normal at a step can
  /// point along a direction with material behind it), and a ray restarted
  /// inside the solid goes on to adsorb onto BURIED cells.
  bool reemitFrom(const viennacs::VoxelHit<NumericType, D> &h,
                  std::array<NumericType, D> &origin,
                  std::array<NumericType, D> &direction) {
    std::array<NumericType, D> n3{}, nd{};
    NumericType len = 0;
    for (int d = 0; d < D; ++d) len += h.normal[d] * h.normal[d];
    len = len > 0 ? std::sqrt(len) : NumericType(1);
    for (int d = 0; d < D; ++d) n3[d] = h.normal[d] / len;
    // cosine-weighted about the normal: a uniform point on the unit sphere
    // (circle in 2D) added to the normal, renormalised -- the same
    // construction viennaray's ReflectionDiffuse uses, on this RNG stream
    for (int tries = 0;; ++tries) {
      std::array<NumericType, D> r{};
      if constexpr (D == 2) {
        const NumericType a = 2 * NumericType(M_PI) * uni();
        r[0] = std::cos(a); r[1] = std::sin(a);
      } else {
        const NumericType z = 2 * uni() - 1;
        const NumericType a = 2 * NumericType(M_PI) * uni();
        const NumericType rad = std::sqrt(std::max(NumericType(0), 1 - z * z));
        r[0] = rad * std::cos(a); r[1] = rad * std::sin(a); r[2] = z;
      }
      NumericType m = 0;
      for (int d = 0; d < D; ++d) { nd[d] = r[d] + n3[d]; m += nd[d] * nd[d]; }
      if (m > NumericType(1e-8)) {
        m = std::sqrt(m);
        for (int d = 0; d < D; ++d) nd[d] /= m;
        break;
      }
      if (tries > 4) { nd = n3; break; }   // degenerate: leave along the normal
    }
    int axis = 0;
    NumericType steepest = 0;
    for (int d = 0; d < D; ++d)
      if (std::abs(n3[d]) > steepest) { steepest = std::abs(n3[d]); axis = d; }
    const int outward = n3[axis] > 0 ? 1 : -1;
    int clear = 0;
    bool escaped = false;
    auto probe = h.index;
    for (int step = 0; step < 8; ++step) {
      probe[axis] += outward;
      ++clear;
      const int nid = lattice_->cellId(probe);
      if (nid < 0 || (*fill_)[nid] <= NumericType(1e-9)) { escaped = true; break; }
    }
    if (!escaped)
      return false;   // still inside the material: drop the particle
    for (int d = 0; d < D; ++d) {
      origin[d] = h.point[d] +
                  n3[d] * delta() * (static_cast<NumericType>(clear) + 1e-3);
      direction[d] = nd[d];
    }
    return true;
  }

  /// The same placement, keeping the caller's direction: what an ion needs
  /// after it reflects.
  bool placeOutside(const viennacs::VoxelHit<NumericType, D> &h,
                    std::array<NumericType, D> &origin) {
    std::array<NumericType, D> keep{}, dummy{};
    for (int d = 0; d < D; ++d) keep[d] = 0;
    if (!reemitFrom(h, origin, dummy))
      return false;
    (void)keep;
    return true;
  }

  /// unit outward normal at a hit
  std::array<NumericType, D>
  unitNormal(const viennacs::VoxelHit<NumericType, D> &h) const {
    std::array<NumericType, D> n{};
    NumericType len = 0;
    for (int d = 0; d < D; ++d) len += h.normal[d] * h.normal[d];
    len = len > 0 ? std::sqrt(len) : NumericType(1);
    for (int d = 0; d < D; ++d) n[d] = h.normal[d] / len;
    return n;
  }

  /// an expected count becomes an integer: floor, plus one with the remainder
  int drawCount(NumericType n) {
    const int k = static_cast<int>(std::floor(n));
    return k + (uni() < n - static_cast<NumericType>(k) ? 1 : 0);
  }

  /// Remove `count` cells from the SURFACE inside the cascade footprint.
  ///
  /// Drilling straight down from the impact bores needles: the column that
  /// was hit runs away from its neighbours, the staircase grows side faces
  /// that rays reach poorly, and the area-averaged coverage falls below what
  /// the flux actually sees. Sputtered material leaves from the exposed
  /// surface, so each removal takes the topmost solid cell of a column drawn
  /// from the footprint.
  ///
  /// The footprint is not free: an ion consumes 2Y/sigma0 of surface, so
  /// R = sqrt(2Y/(sigma0*pi)) -- about 1.7 nm here, the scale of a 100 eV
  /// cascade.
  /// flat index of a column, i.e. of the lateral indices alone
  size_t columnOf(const std::array<int, D> &idx) const {
    const auto &dims = lattice_->dims();
    size_t flat = 0, stride = 1;
    for (int d = 0; d < D - 1; ++d) {
      flat += static_cast<size_t>(idx[d]) * stride;
      stride *= static_cast<size_t>(dims[d]);
    }
    return flat;
  }

  /// a column drawn uniformly from the cascade footprint around `idx`
  std::array<int, D> drawColumn(const std::array<int, D> &idx,
                                NumericType radiusCells) {
    auto col = idx;
    if (radiusCells > 0) {
      const auto &dims = lattice_->dims();
      for (int d = 0; d < D - 1; ++d) {
        const int off =
            static_cast<int>(std::lround((2 * uni() - 1) * radiusCells));
        col[d] = std::clamp(idx[d] + off, 0, dims[d] - 1);
      }
    }
    return col;
  }

  /// Height of the topmost cell of a column that may actually be removed:
  /// solid, not mask, and EXPOSED. Exposure is the point. A column under the
  /// mask still has silicon at its top, but that silicon is buried, and
  /// taking it undercuts the mask -- the trench then grows sideways beneath
  /// it, which the level set never does. Returns -1 when there is nothing to
  /// take.
  int topRemovable(const std::array<int, D> &col) const {
    const auto &dims = lattice_->dims();
    std::array<int, D> at = col;
    for (int k = dims[D - 1] - 1; k >= 0; --k) {
      at[D - 1] = k;
      const int id = lattice_->cellId(at);
      if (id >= 0 && solid(id) && !isMask(id))
        return isExposed(at) ? k : -1;  // buried: nothing below is freer
    }
    return -1;
  }

  /// Remove one exposed, non-mask cell from within `radiusCells` of the
  /// IMPACT, chosen uniformly by reservoir sampling.
  ///
  /// Not the topmost cell of a drawn column. On a sidewall the topmost cell
  /// of a column sits up at the mask, tens of cells above the impact, and
  /// taking it carves a taper into the wall: the trench widens under the mask
  /// while the floor is still descending. Sputtered material leaves from the
  /// surface AT the cascade, so the cell taken has to be near the impact.
  bool removeNearby(const std::array<int, D> &idx, NumericType radiusCells) {
    const int R = std::max(1, static_cast<int>(std::lround(radiusCells)));
    const NumericType R2 = std::max(NumericType(1), radiusCells * radiusCells);
    const auto &dims = lattice_->dims();
    std::array<int, D> lo{}, hi{}, at{}, pick{};
    for (int d = 0; d < D; ++d) {
      lo[d] = std::max(0, idx[d] - R);
      hi[d] = std::min(dims[d] - 1, idx[d] + R);
      at[d] = lo[d];
    }
    int seen = 0;
    while (true) {
      NumericType r2 = 0;
      for (int d = 0; d < D; ++d) {
        const NumericType q = static_cast<NumericType>(at[d] - idx[d]);
        r2 += q * q;
      }
      if (r2 <= R2) {
        const int id = lattice_->cellId(at);
        if (id >= 0 && solid(id) && !isMask(id) && isExposed(at)) {
          ++seen;
          if (uni() * seen < NumericType(1))
            pick = at;                       // reservoir sample of size 1
        }
      }
      int d = 0;
      for (; d < D; ++d) {
        if (++at[d] <= hi[d])
          break;
        at[d] = lo[d];
      }
      if (d == D)
        break;
    }
    if (seen == 0)
      return false;
    removeCellAt(pick);
    return true;
  }

  size_t *removalTally_ = nullptr;

  void removeCellAt(const std::array<int, D> &at_) {
    if (freeze_) return;
    if (removalTally_) ++(*removalTally_);
    if (removalTally_ == &remIE && !histRem.empty()) ++histRem[at_[0]];
    if (removalTally_ == &remTh && !histTh.empty()) ++histTh[at_[0]];
    std::array<int, D> at = at_;
    // Which neighbours are ALREADY surface, before this cell goes. One that
    // is keeps its adsorbate: taking a cell does not scrub the fluorine off
    // the surface around it. Only a neighbour that was buried is newly
    // uncovered, and that one is fresh silicon.
    const std::uint8_t carried = state_[lattice_->cellId(at)] ;
    std::array<bool, 2 * D> wasExposed{};
    int q = 0;
    for (int d = 0; d < D; ++d)
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        auto nb = at;
        nb[d] += sgn;
        wasExposed[q++] = isExposed(nb);
      }
    const int id = lattice_->cellId(at);
    (*fill_)[id] = NumericType(0);
    state_[id] = Bare;
    if (siteCounts_) { nF_[id] = 0; nO_[id] = 0; }
    ++removedCells_;
    q = 0;
    for (int d = 0; d < D; ++d)
      for (int sgn = -1; sgn <= 1; sgn += 2) {
        auto nb = at;
        nb[d] += sgn;
        const int bid = lattice_->cellId(nb);
        if (bid >= 0 && solid(bid) && !isMask(bid) && !wasExposed[q]) {
          // The receding surface CARRIES its adsorbate. Setting the newly
          // uncovered cell to Bare instead annihilates the nu sites of F
          // that sat on the cell just removed: that fluorine never reacts,
          // so it frees sites, drives more adsorption, and the etch runs
          // fast. handDown_ passes the state down instead.
          state_[bid] = handDown_ ? carried : Bare;
          if (siteCounts_) { nF_[bid] = 0; nO_[bid] = 0; }
          ++nBareReset;
        }
        ++q;
      }
  }

  /// Turn one gas cell into film. The material is the film's own, so a cell
  /// grown over the mask is film and etches, reflects and prunes as film.
  void addCellAt(const std::array<int, D> &at, int grownOn) {
    const int id = lattice_->cellId(at);
    if (id < 0)
      return;
    (*fill_)[id] = NumericType(1);
    state_[id] = Bare;
    (*material_)[id] =
        filmMaterial_ >= 0
            ? filmMaterial_
            : (grownOn >= 0 && !isMask(grownOn) ? (*material_)[grownOn]
                                                : static_cast<int>(Material::Si));
    ++depositedCells_;
  }

  /// One stuck precursor. Its atoms go into the gas cell the ray crossed to
  /// reach the surface -- the cell that is about to become film -- and that
  /// cell flips when it holds a whole cell's worth.
  ///
  /// The remainder is HANDED OUTWARD: a cell that has just filled passes what
  /// it holds over to whatever grows on top of it. Without that, every new
  /// layer starts its accumulator at zero and the film lags the dose by half
  /// a cell per layer, which is a growth rate short by dx/(2*thickness) at
  /// every thickness. Nothing is invented and nothing is discarded; the only
  /// material not yet visible is the fraction in flight on the top cell.
  bool depositAt(const viennacs::VoxelHit<NumericType, D> &h,
                 const std::array<NumericType, D> &dir) {
    ++nDepStick;
    std::array<int, D> t = h.index;
    if (h.enteredAxis >= 0 && h.enteredAxis < D) {
      t[h.enteredAxis] += h.enteredSign;
    } else {
      const auto n = interaction_.normalAt(h.index);
      int axis = 0;
      NumericType best = 0;
      for (int d = 0; d < D; ++d)
        if (std::abs(n[d]) > best) {
          best = std::abs(n[d]);
          axis = d;
        }
      t[axis] += n[axis] > 0 ? 1 : -1;
    }
    int tid = lattice_->cellId(t);
    // The face the ray crossed is not always open: a hit accepted after a
    // pass-through, or one taken on a cell tucked behind a ledge, can be
    // entered through a face whose neighbour is already solid. Take the
    // cell's own gas neighbours instead, the one facing back along the ray
    // first, and only give up when the cell has none at all.
    if (tid < 0 || solid(tid)) {
      int best = -1;
      NumericType bestDot = -std::numeric_limits<NumericType>::max();
      for (int d = 0; d < D; ++d)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          auto nb = h.index;
          nb[d] += sgn;
          const int nid = lattice_->cellId(nb);
          if (nid < 0 || solid(nid))
            continue;
          const NumericType dot = -dir[d] * static_cast<NumericType>(sgn);
          if (dot > bestDot) { bestDot = dot; best = nid; t = nb; }
        }
      if (best < 0) {
        ++nDepFail;
        return false;
      }
      tid = best;
    }
    if (depCredit_[tid] <= NumericType(0) && h.cellId >= 0) {
      depCredit_[tid] = depCredit_[h.cellId];   // the layer below hands over
      depCredit_[h.cellId] = 0;
    }
    const NumericType atomsPerCell = p_.rho * std::pow(delta(), D);
    depCredit_[tid] += depAtoms_;
    if (depCredit_[tid] >= atomsPerCell) {
      depCredit_[tid] -= atomsPerCell;
      addCellAt(t, h.cellId);
    }
    return true;
  }

  void removeCells(const std::array<int, D> &idx, int count,
                   NumericType radiusCells = 0) {
    for (int c = 0; c < count; ++c)
      removeNearby(idx, radiusCells);
  }

  /// The fractional path: credit the drawn column with `atoms` and flip cells
  /// only once a whole cell's worth has accumulated.
  void removeAtoms(const std::array<int, D> &idx, NumericType atoms,
                   NumericType radiusCells = 0) {
    if (atoms <= NumericType(0))
      return;
    const size_t c = columnOf(idx);
    const NumericType atomsPerCell = p_.rho * std::pow(delta(), D);
    atomsOwed += atoms;
    credit_[c] += atoms;
    while (credit_[c] >= atomsPerCell) {
      if (!removeNearby(idx, radiusCells)) {
        ++nRemoveFail;
        break;   // nothing left to take: keep the credit, do not discard it
      }
      credit_[c] -= atomsPerCell;
    }
  }

  /// flip `count` cells out of `from` back to Bare, scanning the surface
  /// Clear adsorbate worth `sites`, taking whole cells and debiting each by
  /// the nu*sum|n| sites it actually stands for.
  void clearSites(std::uint8_t from, NumericType sites) {
    int count = drawCount(sites / nuOf({}));
    if (count <= 0)
      return;
    const auto &dims = lattice_->dims();
    size_t nCell = 1;
    for (int d = 0; d < D; ++d)
      nCell *= static_cast<size_t>(dims[d]);
    // a stride walk from a random offset, so the same cells are not always
    // taken first
    const size_t stride = 7919u;
    size_t at = static_cast<size_t>(uni() * static_cast<NumericType>(nCell));
    for (size_t seen = 0; seen < nCell && count > 0; ++seen) {
      at = (at + stride) % nCell;
      std::array<int, D> idx{};
      size_t rem = at;
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lattice_->cellId(idx);
      if (id >= 0 && state_[id] == from && isSurface(idx)) {
        state_[id] = Bare;
        --count;
      }
    }
  }

  /// The same count, drawn NEAREST FIRST: a box around the impact, doubled
  /// until the count is met. The mean is therefore unchanged -- only which
  /// cells give up their adsorbate. Falls back to the whole lattice if the
  /// neighbourhood runs dry, so nothing is clipped.
  /// Sweep the cascade and clear EVERY cell of `from` inside it.
  ///
  /// The continuum sink is 2*Y_ie*J_i*theta_F: proportional to coverage. The
  /// old form conditioned on the ion landing on an F cell AND THEN cleared a
  /// fixed 2Y sites, which double-counts theta_F and, worse, asks for 2Y/nu
  /// fluorinated cells inside a patch that only holds theta_F of that many --
  /// so it silently cleared ~35 % short and the floor ran at theta_F 0.455
  /// against 0.387. Sweeping the patch instead clears
  /// area*sigma0*theta_F = 2*Y*theta_F sites in expectation, which is the
  /// continuum term exactly, and it is self-limiting: it can only take the
  /// fluorine that is actually there.
  void clearSwept(std::uint8_t from, NumericType sites,
                  const std::array<int, D> &centre) {
    if (sites <= 0)
      return;
    // The cascade covers an AREA of surface, sites/sigma0. Rounding that to a
    // whole-cell radius is hopeless when it is smaller than a cell -- Y_p/M
    // asks for 0.51 nm^2 and a 3x3 box sweeps ~3 nm^2, clearing ten times too
    // much oxygen. So take the box, measure the surface area actually inside
    // it, and clear each candidate with probability area_wanted/area_inside.
    // Expected cleared = sites*theta_from either way, and it no longer
    // depends on how the radius rounds.
    const NumericType want = sites / p_.sigma0;          // nm^(D-1)
    const int R = std::max(1, static_cast<int>(
                                  std::ceil(cascadeRadius(sites))));
    const auto &dims = lattice_->dims();
    std::array<int, D> lo{}, hi{}, at{};
    for (int d = 0; d < D; ++d) {
      lo[d] = std::max(0, centre[d] - R);
      hi[d] = std::min(dims[d] - 1, centre[d] + R);
      at[d] = lo[d];
    }
    NumericType inside = 0;
    std::vector<int> cand;
    const NumericType cellA = std::pow(delta(), D - 1);
    while (true) {
      const int id = lattice_->cellId(at);
      if (id >= 0 && solid(id) && isSurface(at)) {
        inside += cellA * areaFactor(at);
        if (state_[id] == from)
          cand.push_back(id);
      }
      int d = 0;
      for (; d < D; ++d) {
        if (++at[d] <= hi[d]) break;
        at[d] = lo[d];
      }
      if (d == D) break;
    }
    if (inside <= 0)
      return;
    const NumericType p = std::min(NumericType(1), want / inside);
    for (int id : cand)
      if (uni() < p)
        state_[id] = Bare;
  }


  void clearSitesNear(std::uint8_t from, NumericType sites,
                      const std::array<int, D> &centre) {
    int count = drawCount(sites / nuOf(centre));
    if (count <= 0)
      return;
    // ONE pass, at the cascade radius. Growing the box until the count is met
    // -- which is what this did -- turns it back into a global walk whenever
    // the neighbourhood is short of that state, and for oxygen it almost
    // always is. An ion can only clear what lies in its own cascade; if that
    // is less than the nominal count, it clears less.
    const int R = std::max(1, static_cast<int>(std::lround(
                                  cascadeRadius(sites))));
    const auto &dims = lattice_->dims();
    std::array<int, D> lo{}, hi{}, at{};
    for (int d = 0; d < D; ++d) {
      lo[d] = std::max(0, centre[d] - R);
      hi[d] = std::min(dims[d] - 1, centre[d] + R);
      at[d] = lo[d];
    }
    std::vector<int> cand;
    while (true) {
      const int id = lattice_->cellId(at);
      if (id >= 0 && state_[id] == from && isSurface(at))
        cand.push_back(id);
      int d = 0;
      for (; d < D; ++d) {
        if (++at[d] <= hi[d])
          break;
        at[d] = lo[d];
      }
      if (d == D)
        break;
    }
    for (size_t i = 0; i < cand.size() && count > 0; ++i) {
      const size_t j =
          i + static_cast<size_t>(uni() * static_cast<NumericType>(cand.size() - i));
      std::swap(cand[i], cand[j < cand.size() ? j : cand.size() - 1]);
      state_[cand[i]] = Bare;
      --count;
    }
  }



  NumericType yieldSputter(NumericType E, NumericType cosT) const {
    const NumericType s2 = std::max(NumericType(0), 1 - cosT * cosT);
    const NumericType f = std::max(NumericType(0), (1 + p_.B_sp * s2) * cosT);
    return p_.A_sp * std::max(NumericType(0), std::sqrt(E) - std::sqrt(p_.Eth_sp)) * f;
  }
  NumericType fIon(NumericType cosT) const {
    const NumericType th = std::acos(std::min(NumericType(1), std::max(NumericType(-1), cosT)));
    if (th <= NumericType(M_PI) / 3)
      return 1;
    return std::max(NumericType(0), 3 - 6 * th / NumericType(M_PI));
  }
  NumericType yieldEnhanced(NumericType E, NumericType cosT) const {
    return p_.A_ie * std::max(NumericType(0), std::sqrt(E) - std::sqrt(p_.Eth_ie)) * fIon(cosT);
  }
  NumericType yieldClearO(NumericType E, NumericType cosT) const {
    return p_.A_p * std::max(NumericType(0), std::sqrt(E) - std::sqrt(p_.Eth_p)) * fIon(cosT);
  }

  /// origin on the top face, direction by the cosine law -- sampled in 3D and
  /// projected in 2D, exactly as the filling-fraction arm's source does
  void sampleRay(NumericType power, std::array<NumericType, D> &origin,
                 std::array<NumericType, D> &direction) {
    const auto &dims = lattice_->dims();
    const auto &lo = lattice_->minCorner();
    for (int d = 0; d < D - 1; ++d)
      origin[d] = lo[d] + delta() * static_cast<NumericType>(dims[d]) * uni();
    origin[D - 1] = lo[D - 1] + delta() * static_cast<NumericType>(dims[D - 1]);
    const NumericType cosT = std::pow(uni(), NumericType(1) / (power + 1));
    const NumericType sinT = std::sqrt(std::max(NumericType(0), 1 - cosT * cosT));
    const NumericType phi = 2 * NumericType(M_PI) * uni();
    if constexpr (D == 2) {
      const NumericType dx = std::cos(phi) * sinT, dy = -cosT;
      const NumericType nr = std::sqrt(dx * dx + dy * dy);
      direction[0] = dx / nr;
      direction[1] = dy / nr;
    } else {
      direction[0] = std::cos(phi) * sinT;
      direction[1] = std::sin(phi) * sinT;
      direction[2] = -cosT;
    }
  }

public:
  VoxelPMC(const viennacs::LatticeMap<NumericType, D> &lattice,
           std::vector<NumericType> &fill, std::vector<int> &material,
           const Parameters &p = Parameters{})
      : lattice_(&lattice), fill_(&fill), material_(&material),
        interaction_(lattice, fill,
                     viennacs::NormalEstimator::InterfaceFit),
        p_(p) {
    // InterfaceAverage, radius 3, by default. Measured on a W=40 nm trench,
    // 30 nm etch, 3 seeds, against the level set: volume removed +4.1 % with
    // this estimator, +13.1 % with Youngs' 3^D stencil, +27.4 % with the face
    // normal. The face normal is not an estimate at all -- a downward ray
    // always enters through a top face, so every ion reads normal incidence.
    // The stencil is a PHYSICAL length, not a cell count. Fixing it at 3
    // cells shrinks it to 1.5 nm at dx = 0.5, the normals collapse back
    // toward face normals, the acceptance test stops rejecting the ledges it
    // exists to reject, and the staircase over-collects again -- refining the
    // grid then makes the answer worse, which is how this was found.
    // Same traversal as the filling-fraction arm: the Embree BVH over the
    // cell boxes, not the grid walk. Both arms must trace identically or the
    // comparison measures the tracer as well as the representation.
    interaction_.setTraversalEngine(viennacs::TraversalEngine::EmbreeBVH);
    interaction_.setInterfaceRadius(fitRadius_);
    state_.assign(fill.size(), Bare);
    // binary from the outset: a fractional cell has no meaning here
    for (auto &f : *fill_)
      f = f >= NumericType(0.5) ? NumericType(1) : NumericType(0);
    size_t columns = 1;
    for (int d = 0; d < D - 1; ++d)
      columns *= static_cast<size_t>(lattice.dims()[d]);
    // Seed each column's accumulator uniformly on [0, atomsPerCell) rather
    // than at zero.
    //
    // Starting at zero makes every column wait for a WHOLE cell's worth of
    // yield before it may give up its first cell, so at any instant the
    // surface owes about half a cell per column that it has not yet paid.
    // That is a systematic deficit of dx/2 in etched depth -- not noise, and
    // not something that averages out, because the credit only ever climbs
    // from zero. Measured on a blanket at dx = 1 etched 10 nm it is 0.47 nm
    // in 2D and 0.46 in 3D, i.e. -4.7 %, and it scales as dx / (2 * depth).
    //
    // Seeding the credit at its own stationary distribution removes the bias:
    // the expected outstanding credit is then the same at the start as at the
    // end, so what is removed over a run equals what was owed over it.
    // Measured on the blanket, 3 seeds, 10 nm at dx = 1:
    //
    //              blanket error        ledger, owed -> taken
    //   2D       -1.2 % -> +2.1 %        21.23 -> 21.05
    //   3D       -7.6 % -> -3.1 %        38.79 -> 38.94
    //
    // 2D gets WORSE and is nonetheless right: starting from zero, its yields
    // over-asking by +4.1 % was cancelled by the -4.6 % deferral, and that
    // cancellation held only at this depth and grid spacing.
    const NumericType atomsPerCell = p_.rho * std::pow(delta(), D);
    credit_.resize(columns);
    for (auto &c : credit_)
      c = static_cast<NumericType>(
              std::uniform_real_distribution<double>(0.0, 1.0)(rng_)) *
          atomsPerCell;
  }

  /// Carry the remainder of a partial cell instead of rounding it away.
  void setFractionalRemoval(bool on) { fractional_ = on; }
  /// Split each ion into `w` super-particles. Needs fractional removal.
  void setIonWeight(NumericType w) { ionWeight_ = w; }
  /// Consume adsorbate from the impact neighbourhood rather than globally.
  void setLocalClearing(bool on) { localClear_ = on; }
  /// Re-emit non-sticking neutrals diffusely, as the continuum arms do.
  void setReemission(bool on) { reemit_ = on; }
  void setFitRadius(int r) { fitRadius_ = r > 1 ? r : 2;
                             interaction_.setInterfaceRadius(fitRadius_); }
  void setReflectRadius(int r) { reflectRadius_ = r > 1 ? r : 2; }
  /// Let the normal-fit stencil shrink itself where the surface is curved.
  /// Off reproduces MCFPM's plain fixed-radius fit.
  void setCurvatureCap(bool on) { interaction_.setCurvatureCap(on); }
  /// Fit the normal over one face only, never across a corner.
  void setSegmentedFit(bool on) { interaction_.setSegmentedFit(on); }
  /// How far off the seed plane a cell may sit and still enter the fit,
  /// in cells. Small keeps the fit on one face; large smooths across a
  /// rounded corner. Off entirely at very large values.
  void setSegmentTolerance(NumericType t) {
    interaction_.setSegmentTolerance(t); }
  void setCurvatureAlpha(NumericType a) { interaction_.setCurvatureAlpha(a); }
  void setMinFitRadius(int r) { interaction_.setMinInterfaceRadius(r); }
  void capStats(long &f, long &fired, double &meanR) const {
    interaction_.capStats(f, fired, meanR); }
  /// Hold the geometry still: chemistry runs, nothing is removed.
  void setFreezeSurface(bool on) { freeze_ = on; }
  /// Off drops rays that leave the side, as before.
  void setSideReflection(bool on) { sideReflect_ = on; }
  /// Normalise a cell's dose by the number of faces it exposes.
  void setAreaNormalisation(int mode) { areaNorm_ = mode; }
  /// Credit the spontaneous removal at the F adsorption event.
  void setFluorineBalance(bool on) { fBalance_ = on; }
  /// Let the receding surface carry its adsorbate downward.
  void setHandDown(bool on) { handDown_ = on; }
  /// Resolve each cell's chemistry to its nu sites instead of one flag.
  /// One particle, one cell, one decision.
  void setSimpleFlux(bool on) { simpleFlux_ = on; }
  /// Set the per-hit removal probability directly: no sticking
  /// coefficient, no desorption, no coverage. Pure transport test.
  void setSimpleP(NumericType p) { simpleP_ = p; }
  void setArmAfter(NumericType cells) { armAfter_ = cells; }
  void setMaxBounce(int n) { kMaxBounce = n > 0 ? n : 1; }
  /// Trace against the reconstructed surface, not the cell faces.
  void setPlaneAcceptance(bool on) { planeAccept_ = on; }
  void setPlaneWindow(NumericType cells) { planeWindow_ = cells; }
  void setMinFitPoints(int n) { minFitPts_ = n; }
  /// Whether the caller intends to prune after the run; the PMC never
  /// prunes on its own, since doing so mid-run alters the etch.
  void setPruneIslands(bool on) { prune_ = on; }
  bool pruneIslands() const { return prune_; }
  void setOffPlaneTest(bool on) { offPlaneTest_ = on; }

  /// Conformal deposition instead of etching: the surface grows.
  void setDeposition(bool on) {
    deposit_ = on;
    if (!on)
      return;
    depCredit_.assign(fill_->size(), NumericType(0));
    // Seed the accumulator on the cells that are ALREADY surface, uniformly
    // on [0, atomsPerCell). Starting every one at zero makes the first layer
    // wait for a whole cell's worth of atoms before any of it appears, and
    // the hand-over carries that lag through every layer after it, so the
    // film stays half a cell thin for the whole run. Seeding at the
    // stationary distribution pays the offset once, at t = 0 -- the same
    // argument as the etch's column credit.
    const NumericType atomsPerCell = p_.rho * std::pow(delta(), D);
    const auto &dims = lattice_->dims();
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);
    std::array<int, D> idx{};
    for (size_t flat = 0; flat < sites; ++flat) {
      size_t rem = flat;
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      const int id = lattice_->cellId(idx);
      if (id >= 0 && solid(id) && isExposed(idx))
        depCredit_[id] = uni() * atomsPerCell;
    }
  }
  /// net sticking probability per surface hit
  void setDepositionP(NumericType p) { depP_ = p; }
  /// solid atoms one stuck molecule carries
  void setDepositionAtoms(NumericType a) { depAtoms_ = a; }
  void setFilmMaterial(int m) { filmMaterial_ = m; }
  bool deposition() const { return deposit_; }
  size_t depositedCells() const { return depositedCells_; }
  /// removal probability per hit, set so ONE hit per particle reproduces
  /// the analytic blanket rate: s0*(1-theta)/4 atoms, over rho*dx^D per cell
  NumericType simpleRemoveP() const {
    if (simpleP_ >= 0)
      return simpleP_;
    const NumericType a = p_.fluxF * p_.stickF;
    const NumericType th = a / (a + 4 * p_.kSigma);
    return p_.stickF * (1 - th) / 4 / (p_.rho * std::pow(delta(), D));
  }
  void setFluxEtch(bool on) {
    fluxEtch_ = on;
    if (on) arrF_.assign(state_.size(), 0u);
  }
  void setSiteCounts(bool on) {
    siteCounts_ = on;
    if (on) { nF_.assign(state_.size(), 0); nO_.assign(state_.size(), 0); }
  }
  /// Which normal a ray sees. Face gives the axis-aligned facet it entered
  /// through, so on a staircase the incidence angle is quantised and a
  /// vertical facet reads cos(theta) = 0 -- the ion yield's angular factor
  /// then vanishes there, and a re-emitted neutral leaves horizontally.
  /// FillGradientYoungs is what the filling-fraction arm uses, and smooths
  /// the staircase over its stencil.
  void setNormalEstimator(viennacs::NormalEstimator e) {
    interaction_.setNormalEstimator(e);
  }

  /// GridDDA walks the lattice; EmbreeBVH traces the cell boxes, which
  /// is what the filling-fraction arm uses and hence the default here.
  void setTraversalEngine(viennacs::TraversalEngine e) {
    interaction_.setTraversalEngine(e); }
  viennacs::TraversalEngine traversalEngine() const {
    return interaction_.traversalEngine(); }
  void setSeed(unsigned s) { rng_.seed(s); }
  size_t removedCells() const { return removedCells_; }
  /// Atoms still credited to columns and not yet turned into cells.
  double outstandingCredit() const {
    double t = 0;
    for (auto c : credit_) t += static_cast<double>(c);
    return t; }
  const std::vector<std::uint8_t> &states() const { return state_; }

  /// theta_F, theta_O -- counted, never stored
  std::array<NumericType, 2> coverages() const {
    const auto &dims = lattice_->dims();
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);
    size_t nF = 0, nO = 0, n = 0;
    NumericType fSum = 0, oSum = 0;
    std::array<int, D> idx{};
    for (size_t flat = 0; flat < sites; ++flat) {
      size_t rem = flat;
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      if (!isSurface(idx))
        continue;
      const int id = lattice_->cellId(idx);
      ++n;
      if (siteCounts_) {
        const NumericType nu = nuScalar();
        fSum += static_cast<NumericType>(nF_[id]) / nu;
        oSum += static_cast<NumericType>(nO_[id]) / nu;
      } else {
        if (state_[id] == Fluorinated) ++nF;
        else if (state_[id] == Oxidised) ++nO;
      }
    }
    if (!n)
      return {0, 0};
    if (siteCounts_)
      return {fSum / static_cast<NumericType>(n), oSum / static_cast<NumericType>(n)};
    return {static_cast<NumericType>(nF) / static_cast<NumericType>(n),
            static_cast<NumericType>(nO) / static_cast<NumericType>(n)};
  }

  /// SUPPORT. Whole-cell removal leaves islands: a cell whose neighbours all
  /// went before it is still solid, still exposed, and only leaves when a ray
  /// happens to find it. The filling-fraction arm settles this globally --
  /// matter is supported only if it is face-connected to the bulk, and what
  /// is not is removed and counted (csVoxelAdvance, "SUPPORT"). This is the
  /// binary equivalent: flood from the lattice floor and from the mask, then
  /// take whatever the flood did not reach.
  ///
  /// Call it ONCE, after the run. Doing it every step removes each island the
  /// moment it forms, and those cells would otherwise have taken rays some
  /// time to clear -- measured on a W=40 nm trench etched 15 nm, per-step
  /// pruning took 35 cells against the 17 islands actually present at the end
  /// and ran the floor 0.9 nm deep (-16.25 against -15.31). As a final pass
  /// the transport is untouched and only the debris goes.
  size_t pruneUnsupported() {
    const auto &dims = lattice_->dims();
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);
    std::vector<unsigned char> seen(fill_->size(), 0);
    std::vector<std::array<int, D>> stack;
    std::array<int, D> idx{};
    auto unflatten = [&](size_t flat) {
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(flat % static_cast<size_t>(dims[d]));
        flat /= static_cast<size_t>(dims[d]);
      }
    };
    // seeds: any solid cell on the bottom face of the lattice, and the mask
    for (size_t flat = 0; flat < sites; ++flat) {
      unflatten(flat);
      const int id = lattice_->cellId(idx);
      if (id < 0 || !solid(id) || seen[id])
        continue;
      if (idx[D - 1] == 0 || isMask(id)) {
        seen[id] = 1;
        stack.push_back(idx);
      }
    }
    while (!stack.empty()) {
      const auto at = stack.back();
      stack.pop_back();
      for (int d = 0; d < D; ++d)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
          auto nb = at;
          nb[d] += sgn;
          const int nid = lattice_->cellId(nb);
          if (nid >= 0 && solid(nid) && !seen[nid]) {
            seen[nid] = 1;
            stack.push_back(nb);
          }
        }
    }
    size_t pruned = 0;
    for (size_t flat = 0; flat < sites; ++flat) {
      unflatten(flat);
      const int id = lattice_->cellId(idx);
      if (id >= 0 && solid(id) && !isMask(id) && !seen[id]) {
        removeCellAt(idx);
        ++pruned;
      }
    }
    nPruned += pruned;
    return pruned;
  }

  /// One step of `dt`: deliver the particles that arrive in that time, then
  /// fire the thermal events that occur in it.
  void step(NumericType dt) {
    const auto &dims = lattice_->dims();
    NumericType area = 1;
    for (int d = 0; d < D - 1; ++d)
      area *= delta() * static_cast<NumericType>(dims[d]);
    // atoms per cell and adsorption SITES per cell are different counts
    const NumericType atomsPerCell = p_.rho * std::pow(delta(), D);
    const NumericType nu = p_.sigma0 * std::pow(delta(), D - 1);
    interaction_.prepare();

    auto deliver = [&](NumericType flux, NumericType power, bool ion,
                       std::uint8_t adsorbState, NumericType stick) {
      // J is quoted per site, so the areal flux is J*sigma0
      // NB: no kFluxBoost here. A rejected ray PASSES THROUGH to the next
      // cell rather than being absorbed, so the acceptance only redistributes
      // flux between facets -- the total is conserved and needs no
      // compensation. Emitting K times more particles simply added K times
      // the dose (+4.4 % -> +15.0 % volume when tried).
      const NumericType expected = flux * p_.sigma0 * area * dt;
      std::poisson_distribution<long long> pois(
          static_cast<double>(std::max(NumericType(0), expected)));
      const long long n = pois(rng_);
      std::array<NumericType, D> origin{}, direction{};
      for (long long i = 0; i < n; ++i) {
        sampleRay(power, origin, direction);
        if (!ion) ++nLaunch;
        auto o0 = origin, d0 = direction;
        const auto hit = traceReflective(o0, d0);
        origin = o0; direction = d0;
        if (!hit.hit())
          continue;
        if (!ion) {
          // A neutral is absorbed only on a bare cell -- an occupied one has
          // no free site, which is what the factor (1-thF-thO) IS. Absorption
          // fills ONE of the nu sites the cell carries, so the cell itself
          // flips one time in nu. Anything not absorbed reflects.
          auto o = origin, dir = direction;
          auto h = hit;
          for (int bounce = 0;; ++bounce) {
            const int hid = h.cellId;
            if (!resolveImpact(h, dir)) {       // the plane is elsewhere
              if (bounce >= kMaxBounce) break;   // a separate, lattice-scaled
              passThrough(h, dir, o);            // budget measured WORSE: 3D
                                                 // -27.9 % -> -31.9 %
              h = traceReflective(o, dir);
              if (!h.hit() || !isExposed(h.index)) break;
              continue;
            }
            ++nHitN;              // a real surface hit, not a re-emission
            if (deposit_) {
              // stick and be consumed, or reflect. Nothing else happens to
              // the cell that was hit.
              if (uni() < depP_) {
                depositAt(h, dir);
                break;
              }
              // otherwise it reflects, handled by the re-emission below
            } else if (simpleFlux_ && adsorbState == Fluorinated) {
              // stick with s0*(1-theta) and be CONSUMED; a stuck F takes a
              // quarter of a silicon atom with it. Reflection is then free:
              // the same probability applies wherever the particle lands
              // next, so multiple hits do not multiply the removal.
              if (!isMask(hid) && uni() < simpleRemoveP()) {
                removalTally_ = &remTh;
                if (bounce == 0) ++nRem0; else ++nRemB;
                removeCellAt(h.index);        // remove the cell it hit
                break;                        // consumed
              }
              // otherwise it reflects, handled by the re-emission below
            } else if (fluxEtch_ && adsorbState == Fluorinated && !isMask(hid)) {
              ++arrF_[hid];      // the cell's own arrival tally
              break;             // the particle is consumed here
            }
            // The simple/flux modes handle the particle themselves; without
            // this guard the ordinary adsorption ran underneath them and its
            // thermal channel added its own removals on top.
            if (deposit_ || simpleFlux_ || fluxEtch_) {
              // fall through to the re-emission below
            } else if (siteCounts_) {
              // acceptance is the FREE-SITE fraction, not an all-or-nothing
              // test on the cell's flag
              if (!isMask(hid) && uni() < stick * freeFrac(hid)) {
                const NumericType nuS = nuScalar();
                if (adsorbState == Fluorinated) {
                  if (nF_[hid] < static_cast<int>(nuS)) ++nF_[hid];
                  ++nAdsF;
                  if (fBalance_) {
                    removalTally_ = &remTh;
                    removeAtoms(h.index, NumericType(0.25));
                  }
                } else {
                  if (nO_[hid] < static_cast<int>(nuS)) ++nO_[hid];
                  ++nAdsO;
                }
                syncState(hid);
                break;
              }
            } else if (!isMask(hid) && state_[hid] == Bare && uni() < stick) {
              // one arrival fills ONE site of the nu*sum|n| this cell holds
              if (uni() < NumericType(1) / (nu * areaFactor(h.index))) {
                state_[hid] = adsorbState;
                if (adsorbState == Fluorinated) ++nAdsF; else ++nAdsO;
                // FBALANCE: drive the spontaneous etch off the F BALANCE
                // rather than off a per-cell firing rate. A flip to [F]
                // stands for nu adsorbed sites and 4F* + Si -> SiF4 takes
                // four of them per silicon, so credit nu/4 atoms here. The
                // removal is then per impact, like every other channel, and
                // the staircase's own area never enters it.
                if (fBalance_ && adsorbState == Fluorinated) {
                  removalTally_ = &remTh;
                  removeAtoms(h.index, nu / 4);
                }
              }
              break;                        // absorbed onto a site
            }
            if (!reemit_ || bounce >= kMaxBounce)
              break;                        // discarded, as before
            if (bounce + 1 >= kMaxBounce) ++nBounceCap;
            ++nBounce;
            if (!reemitFrom(h, o, dir)) {
              ++nBounceFail;
              break;                        // could not be placed in gas
            }
            h = traceReflective(o, dir, armAfter_ * delta());
            if (!h.hit() || !isExposed(h.index)) {
              ++nEscape;
              break;                        // left the domain, or went inside
            }
          }
          continue;
        }
        // ---- the ion's own path. It deposits at every hit and, above
        // thetaRMin, reflects with a coned-cosine direction and a reduced
        // energy -- the ionSource model the level-set arm uses. Absorbing
        // every ion on first hit throws away exactly the grazing ones that
        // would have carried their yield onto the floor.
        auto o = origin, dir = direction;
        auto h = hit;
        NumericType E = p_.meanEnergy +
                        p_.sigmaEnergy *
                            std::normal_distribution<NumericType>(0, 1)(rng_);
        E = std::max(NumericType(0), E);
        const NumericType minEth =
            std::min(p_.Eth_p, std::min(p_.Eth_sp, p_.Eth_ie));
        const NumericType tMin = p_.thetaRMin * NumericType(M_PI) / 180;
        const NumericType tMax = p_.thetaRMax * NumericType(M_PI) / 180;

        for (int bounce = 0;; ++bounce) {
          if (!resolveImpact(h, dir)) {
            if (bounce >= kMaxBounce) break;
            passThrough(h, dir, o);
            h = traceReflective(o, dir);
            if (!h.hit() || !isExposed(h.index)) break;
            continue;
          }
          const int id = h.cellId;
          const auto n = unitNormal(h);
          NumericType cosT = 0;
          for (int d = 0; d < D; ++d)
            cosT += -dir[d] * n[d];
          cosT = std::min(NumericType(1), std::max(NumericType(0), cosT));
          const NumericType incAngle = std::acos(cosT);

          if (!isMask(id)) {
            {  // bin the hit by the local surface angle
              const int q = std::min(8, (int)(std::acos(std::min(NumericType(1),
                            cosT)) * 180.0 / M_PI / 10.0));
              ++angHit[q];
              // is this a wall cell (exposed sideways but not from above)?
              bool up = false, lat = false;
              for (int d = 0; d < D; ++d)
                for (int sg = -1; sg <= 1; sg += 2) {
                  auto nb = h.index; nb[d] += sg;
                  const int nid = lattice_->cellId(nb);
                  if (nid >= 0 && solid(nid)) continue;
                  if (d == D - 1 && sg > 0) up = true; else lat = true;
                }
              const double y = yieldEnhanced(E, cosT);
              if (lat && !up) { ++wallHit[q]; wallY[q] += y; }
              else            { ++floorHit[q]; floorY[q] += y; }
            }
            sumCosIon += cosT; ++nCosIon;
            ++nIons;
            if (histHit.empty()) {
              histHit.assign(lattice_->dims()[0], 0);
              histOnF.assign(lattice_->dims()[0], 0);
              histRem.assign(lattice_->dims()[0], 0);
              histTh.assign(lattice_->dims()[0], 0);
            }
            ++histHit[h.index[0]];
            if (uni() < fFrac(id)) ++histOnF[h.index[0]];
            const NumericType Ysp = yieldSputter(E, cosT);
            removalTally_ = &remSp;
            const NumericType Rsp = damageRadius(Ysp);
            if (fractional_)
              removeAtoms(h.index, Ysp / ionWeight_, Rsp);
            else
              removeCells(h.index, drawCount(Ysp / atomsPerCell), Rsp);

            if (uni() < fFrac(id)) {
              ++nIonsOnF;
              {
                if (reflRem.empty()) {
                  reflRem.assign(lattice_->dims()[0], 0);
                  directRem.assign(lattice_->dims()[0], 0);
                }
                const double y = yieldEnhanced(E, cosT);
                const int q = std::min(8, (int)(std::acos(std::min(
                    NumericType(1), cosT)) * 180.0 / M_PI / 10.0));
                if (bounce > 0) { ++reflRem[h.index[0]]; reflY += y; ++reflAng[q]; }
                else            { ++directRem[h.index[0]]; directY += y; }
              }
              { const int q = std::min(8, (int)(std::acos(std::min(NumericType(1),
                              cosT)) * 180.0 / M_PI / 10.0));
                ++angRem[q]; }
              removalTally_ = &remIE;
              const NumericType Y = yieldEnhanced(E, cosT);
              const NumericType R = damageRadius(Y);
              if (fractional_)
                removeAtoms(h.index, Y / ionWeight_, R);
              else
                removeCells(h.index, drawCount(Y / atomsPerCell), R);
              // NOTE: the sweep is done once per ion below, not here -- it
              // must not be conditioned on this cell being F as well.
            }
            // the cascade sweeps the surface it covers, whatever it landed on
            { const NumericType sF = 2 * yieldEnhanced(E, cosT) / ionWeight_;
              nClearF += (size_t)(sF / nu);
              clearSwept(Fluorinated, sF, h.index);
              clearSwept(Oxidised, yieldClearO(E, cosT) / ionWeight_, h.index); }
          }

          if (bounce == 0) ++bounceHist[0];
          // ---- reflect or stop, exactly as ChemicalIon::surfaceReflection
          NumericType sticking = 1;
          if (incAngle > tMin)
            sticking = 1 - std::min(NumericType(1),
                                    std::max(NumericType(0),
                                             (incAngle - tMin) / (tMax - tMin)));
          if (sticking >= 1 || uni() < sticking || bounce >= kMaxBounce)
            break;
          const NumericType newE = reflectedEnergy(E, incAngle);
          if (newE <= minEth)
            break;
          E = newE;
          ++nIonBounce;
          if (bounce + 1 < 12) ++bounceHist[bounce + 1];
          // specular direction off the SMOOTHED wall, not the local facet
          auto nRef = n;
          {
            const auto nw = interaction_.fitNormalAt(h.index, reflectRadius_);
            NumericType l = 0;
            for (int d = 0; d < D; ++d) l += nw[d] * nw[d];
            if (l > NumericType(0.5))
              for (int d = 0; d < D; ++d) nRef[d] = nw[d];
          }
          const auto nd = reflectConed(
              dir, nRef,
              NumericType(M_PI_2) -
                  std::min(incAngle, p_.minAngle * NumericType(M_PI) / 180));
          if (!placeOutside(h, o))
            break;
          dir = nd;
          h = traceReflective(o, dir);
          if (!h.hit() || !isExposed(h.index))
            break;
        }
      }
    };

    if (deposit_) {
      // one precursor species, and nothing that fires with time
      deliver(p_.fluxF, p_.cosinePowerNeutral, false, Fluorinated, depP_);
      return;
    }
    deliver(p_.fluxF, p_.cosinePowerNeutral, false, Fluorinated, p_.stickF);
    deliver(p_.fluxO, p_.cosinePowerNeutral, false, Oxidised, p_.stickO);
    deliver(p_.fluxIon * ionWeight_, p_.cosinePowerIon, true, Bare, 0);

    // ---- thermal events, per unit TIME and per unit AREA
    //
    // These rates are areal: k_sigma is a velocity per unit of TRUE surface.
    // Firing once per exposed CELL charges the staircase's own area instead,
    // and a staircase carries ~1.5x the surface of the shape it represents --
    // measured 269 nm of cell faces against 171 nm of level-set arc on the
    // same trench. Every isotropic channel then over-removes by that factor.
    //
    // A cell whose local interface normal is n carries a true area
    // dx^(D-1) * sum|n_i|. Count it: a 45 deg staircase has ONE exposed cell
    // per sqrt(2) of true surface, so each carries sqrt(2), and an
    // axis-aligned facet carries 1. Weighting the firing by that makes the
    // rate areal again.
    const NumericType pF = 1 - std::exp(-4 * p_.kSigma * dt);
    const NumericType pO = 1 - std::exp(-p_.betaSigma * dt);
    auto areaWeight = [&](const std::array<int, D> &idx) {
      const auto n = interaction_.normalAt(idx);
      NumericType sum = 0;
      for (int d = 0; d < D; ++d)
        sum += std::abs(n[d]);
      return sum > NumericType(1e-6) ? sum : NumericType(1);
    };
    // per F cell the removal rate is k_sigma*sigma0/(rho*dx) against a firing
    // rate of 4*k_sigma, so a firing removes the cell this often
    const NumericType pRemove = p_.sigma0 / (4 * p_.rho * delta());
    size_t sites = 1;
    for (int d = 0; d < D; ++d)
      sites *= static_cast<size_t>(dims[d]);
    std::array<int, D> idx{};
    for (size_t flat = 0; flat < sites; ++flat) {
      size_t rem = flat;
      for (int d = 0; d < D; ++d) {
        idx[d] = static_cast<int>(rem % static_cast<size_t>(dims[d]));
        rem /= static_cast<size_t>(dims[d]);
      }
      if (!isSurface(idx))
        continue;
      const int id = lattice_->cellId(idx);
      const NumericType w = areaWeight(idx);
      // NOTE: the weight is applied to the FIRING RATE here, not to the
      // volume each firing takes. Deriving it the other way round -- a
      // per-site firing rate and an areal volume -- is what the bookkeeping
      // argues for, and it measures WORSE (+10.3 % volume, wall theta_F 0.267
      // against a target 0.085, versus +5.1 % and 0.097 for this form). The
      // derivation is therefore missing something; this form is kept because
      // it is the one that matches, and the discrepancy is unresolved.
      if (fluxEtch_) {
        const NumericType nu = nuScalar();
        // arrivals per SITE per second on this cell
        const NumericType G =
            static_cast<NumericType>(arrF_[id]) / (nu * dt);
        const NumericType a = G * p_.stickF;
        const NumericType th = a / (a + 4 * p_.kSigma);
        arrF_[id] = 0;
        if (th > 0) {
          removalTally_ = &remTh;
          removeAtoms(idx, p_.kSigma * th * nu * dt);
        }
      } else if (siteCounts_) {
        int k = 0;
        for (int i = 0; i < nF_[id]; ++i)
          if (uni() < pF) ++k;
        if (k) {
          nF_[id] -= static_cast<std::uint8_t>(k);
          nThermF += k;
          if (!fBalance_) {
            removalTally_ = &remTh;
            removeAtoms(idx, NumericType(k) / 4);
          }
        }
        int m = 0;
        for (int i = 0; i < nO_[id]; ++i)
          if (uni() < pO) ++m;
        if (m) nO_[id] -= static_cast<std::uint8_t>(m);
        syncState(id);
      } else if (state_[id] == Fluorinated && uni() < pF) {
        state_[id] = Bare;
        ++nThermF;
        if (!fBalance_) {
          removalTally_ = &remTh;
          if (fractional_)
            removeAtoms(idx, w * nu / 4); // 1 Si per 4 F, over nu*w sites
          else if (uni() < w * pRemove)
            removeCells(idx, 1);
        }
      } else if (state_[id] == Oxidised && uni() < pO) {
        state_[id] = Bare;
      }
    }
  }
};

} // namespace viennaps
