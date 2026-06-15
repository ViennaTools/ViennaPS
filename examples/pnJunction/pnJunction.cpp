/**
 * pnJunction.cpp — Lateral PN junction by sequential masked implantation.
 *
 * All simulation parameters are read from config.txt (no hardcoded values).
 * The config file records where every parameter was taken from (modeldb table
 * row, SIMS calibration, or literature), making it straightforward to swap in
 * calibrated values without touching this code.
 *
 * Two implant steps, each with the opposite half of the domain masked:
 *
 *   |<—— P opening (x < 0) ——>|<—— B opening (x > 0) ——>|
 *         P, 30 keV, 1e14 cm⁻²     B, 20 keV, 1e14 cm⁻²
 *   ───────────────────────────────────────────────────────
 *                     Si substrate
 *
 * After a 1000 °C / 30 s anneal the metallurgical PN junction lies near x = 0.
 *
 * Usage: pnJunction [config.txt]
 */

#include "../ionImplantation/exampleConfig.hpp"

#include <psDomain.hpp>
#include <process/psProcess.hpp>
#include <lsMakeGeometry.hpp>
#include <models/psAnneal.hpp>
#include <models/psIonImplantation.hpp>
#include <csNetDoping.hpp>
#include <csSheetResistance.hpp>
#include <vcUtil.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace viennaps;
using T = double;
constexpr int D = 2;


// ── Build substrate with pad oxide ───────────────────────────────────────────
// Stack (bottom to top):
//   Si  : y ∈ [-substrateDepth, 0]
//   SiO2: y ∈ [0, padOxideThickness]      ← pad oxide (screen for implant)
//   Air : y ∈ [padOxideThickness, padOxideThickness + topSpace]
SmartPointer<Domain<T, D>> buildSubstrate(T xExtent, T substrateDepth,
                                          T topSpace, T padOxideThickness,
                                          T gridDelta)
{
    const T domainTop = topSpace + padOxideThickness;
    T bounds[2 * D] = {-0.5 * xExtent,  0.5 * xExtent,
                       -substrateDepth,  domainTop};
    BoundaryType bc[D] = {BoundaryType::REFLECTIVE_BOUNDARY,
                          BoundaryType::INFINITE_BOUNDARY};

    auto domain = Domain<T, D>::New(bounds, bc, gridDelta);
    auto makeLS = [&]() {
        return SmartPointer<viennals::Domain<T, D>>::New(bounds, bc, gridDelta);
    };

    // Si bottom half-space
    {
        auto ls = makeLS();
        T origin[D] = {}, normal[D] = {};
        origin[D - 1] = -substrateDepth;
        normal[D - 1] = T(1);
        viennals::MakeGeometry<T, D>(ls,
            viennals::Plane<T, D>::New(origin, normal)).apply();
        domain->insertNextLevelSetAsMaterial(ls, Material::Si);
    }
    // Si surface at y = 0
    {
        auto ls = makeLS();
        T origin[D] = {}, normal[D] = {};
        normal[D - 1] = T(1);
        viennals::MakeGeometry<T, D>(ls,
            viennals::Plane<T, D>::New(origin, normal)).apply();
        domain->insertNextLevelSetAsMaterial(ls, Material::Si);
    }
    // SiO2 pad oxide top at y = padOxideThickness
    {
        auto ls = makeLS();
        T origin[D] = {}, normal[D] = {};
        origin[D - 1] = padOxideThickness;
        normal[D - 1] = T(1);
        viennals::MakeGeometry<T, D>(ls,
            viennals::Plane<T, D>::New(origin, normal)).apply();
        domain->insertNextLevelSetAsMaterial(ls, Material::SiO2);
    }

    domain->generateCellSet(domainTop, Material::Air, /*isAboveSurface=*/true);
    domain->getCellSet()->buildNeighborhood();
    return domain;
}


// ── Temporarily mask one lateral half of the domain ──────────────────────────
std::vector<T> maskHalf(SmartPointer<Domain<T, D>> domain, bool maskRightHalf)
{
    auto cs  = domain->getCellSet();
    auto *mat = cs->getScalarData("Material");
    std::vector<T> original(mat->begin(), mat->end());

    const T airId  = static_cast<T>(static_cast<int>(Material::Air));
    const T maskId = static_cast<T>(static_cast<int>(Material::Mask));

    for (int i = 0; i < cs->getNumberOfCells(); ++i) {
        if ((*mat)[i] != airId) continue;
        const auto c = cs->getCellCenter(i);
        if (maskRightHalf ? (c[0] > T(0)) : (c[0] < T(0)))
            (*mat)[i] = maskId;
    }
    return original;
}

void restoreMaterial(SmartPointer<Domain<T, D>> domain,
                     const std::vector<T> &original)
{
    auto *mat = domain->getCellSet()->getScalarData("Material");
    std::copy(original.begin(), original.end(), mat->begin());
}


// ── Configure and run one IonImplantation step ────────────────────────────────
void runImplant(SmartPointer<Domain<T, D>> domain,
                const std::string &concLabel, const std::string &damageLabel,
                T rp, T sigma, T skewness, T kurtosis,
                T rpTail, T sigmaTail, T skewnessTail, T kurtTail,
                T headFraction, T dose, T tiltDeg,
                T lateralSigmaHead, T lateralSigmaTail)
{
    PearsonIVParameters<T> head, tail;
    head.mu = rp;       head.sigma = sigma;
    head.beta = skewness; head.gamma = kurtosis;
    tail.mu = rpTail;   tail.sigma = sigmaTail;
    tail.beta = skewnessTail; tail.gamma = kurtTail;

    auto implantModel = SmartPointer<ImplantDualPearsonIV<T, D>>::New(
        head, tail, headFraction,
        T(0), lateralSigmaHead,
        T(0), lateralSigmaTail);

    auto implant = SmartPointer<IonImplantation<T, D>>::New();
    implant->setImplantModel(implantModel);
    implant->setDose(dose);
    implant->setTiltAngle(tiltDeg);
    implant->setLengthUnit(T(1e-7));
    implant->setDoseControl(ImplantDoseControl::WaferDose);
    implant->setMaskMaterials({Material::Mask});
    implant->setScreenMaterials({Material::SiO2});
    implant->setConcentrationLabel(concLabel);
    implant->setDamageLabel(damageLabel);

    Process<T, D>(domain, implant, T(0)).apply();
}


// ── Write a 1-D positive-depth profile to CSV (peak across x per depth slice) ─
void writeDepthProfile(SmartPointer<Domain<T, D>> domain,
                       const std::string &fieldLabel,
                       const std::string &filename)
{
    auto cs    = domain->getCellSet();
    const T delta = cs->getGridDelta();
    auto *data = cs->getScalarData(fieldLabel);
    if (!data) { std::cerr << "Field " << fieldLabel << " not found\n"; return; }

    std::map<T, T> profile;
    for (int i = 0; i < cs->getNumberOfCells(); ++i) {
        const T val = (*data)[i];
        if (val <= T(0)) continue;
        const T depth = -cs->getCellCenter(i)[1];
        if (depth < T(0)) continue;
        const T key = std::round(static_cast<double>(depth / delta)) * delta;
        profile[key] = std::max(profile[key], val);
    }

    std::ofstream f(filename);
    f << "depth_nm,value\n";
    for (const auto &[d, v] : profile)
        f << std::fixed << std::setprecision(3) << d << "," << v << "\n";
    std::cout << "  Wrote " << filename << "\n";
}


// ── Write a lateral profile at fixed positive substrate depth to CSV ─────────
void writeLateralProfile(SmartPointer<Domain<T, D>> domain,
                         const std::string &fieldLabel,
                         T atDepth, const std::string &filename)
{
    auto cs    = domain->getCellSet();
    const T delta = cs->getGridDelta();
    auto *data = cs->getScalarData(fieldLabel);
    if (!data) return;

    std::map<T, std::pair<T, int>> bins;
    for (int i = 0; i < cs->getNumberOfCells(); ++i) {
        const auto center = cs->getCellCenter(i);
        const T depth = -center[1];
        if (depth < T(0)) continue;
        if (std::abs(depth - atDepth) > delta * T(0.6)) continue;
        const T key = std::round(static_cast<double>(center[0] / delta)) * delta;
        bins[key].first  += (*data)[i];
        bins[key].second += 1;
    }

    std::ofstream f(filename);
    f << "x_nm,value\n";
    for (const auto &[x, sv] : bins)
        f << std::fixed << std::setprecision(3) << x << ","
          << sv.first / sv.second << "\n";
    std::cout << "  Wrote " << filename << "\n";
}


// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
    const std::string cfgPath = argc > 1 ? argv[1] : "config.txt";

    util::Parameters params;
    params.readConfigFile(cfgPath);
    if (params.m.empty()) {
        std::cerr << "Config not found: " << cfgPath << "\n";
        std::cerr << "Usage: " << argv[0] << " [config.txt]\n";
        return 1;
    }

    std::cout << "=== Lateral PN junction (config: " << cfgPath << ") ===\n\n";

    // ── Read geometry ─────────────────────────────────────────────────────────
    const T gridDelta      = static_cast<T>(params.get("gridDelta"));
    const T xExtent        = static_cast<T>(params.get("xExtent"));
    const T topSpace          = static_cast<T>(params.get("topSpace"));
    const T substrateDepth    = static_cast<T>(params.get("substrateDepth"));
    const T padOxideThickness = static_cast<T>(params.get("padOxideThickness"));

    // ── Read anneal conditions ────────────────────────────────────────────────
    const T annealTempK = static_cast<T>(params.get("annealTemperatureC")) + T(273.15);
    const T annealTime  = static_cast<T>(params.get("annealTimeS"));

    // ── Read analysis parameters ──────────────────────────────────────────────
    const T scanDepth = std::abs(static_cast<T>(params.get("junctionScanDepthNm")));

    // ── Read P implant parameters ─────────────────────────────────────────────
    const T pDose           = static_cast<T>(params.get("pDoseCm2"));
    const T pTilt           = static_cast<T>(params.get("pTiltDeg"));
    const T pRotation       = static_cast<T>(params.get("pRotationDeg"));
    const T pRp             = static_cast<T>(params.get("pProjectedRange"));
    const T pSigma          = static_cast<T>(params.get("pDepthSigma"));
    const T pSkewness       = static_cast<T>(params.get("pSkewness"));
    const T pKurtosis       = static_cast<T>(params.get("pKurtosis"));
    const T pLatHead        = static_cast<T>(params.get("pLateralSigmaHead"));
    const T pHeadFrac       = static_cast<T>(params.get("pHeadFraction"));
    const T pRpTail         = static_cast<T>(params.get("pTailProjectedRange"));
    const T pSigmaTail      = static_cast<T>(params.get("pTailDepthSigma"));
    const T pSkewnessTail   = static_cast<T>(params.get("pTailSkewness"));
    const T pKurtTail       = static_cast<T>(params.get("pTailKurtosis"));
    const T pLatTail        = static_cast<T>(params.get("pTailLateralSigma"));
    const T pD0             = static_cast<T>(params.get("pAnnealD0"));
    const T pEa             = static_cast<T>(params.get("pAnnealEa"));
    const T pSolC0          = static_cast<T>(params.get("pSolidSolubilityC0"));
    const T pSolEa          = static_cast<T>(params.get("pSolidSolubilityEa"));

    // ── Read B implant parameters ─────────────────────────────────────────────
    const T bDose           = static_cast<T>(params.get("bDoseCm2"));
    const T bTilt           = static_cast<T>(params.get("bTiltDeg"));
    const T bRotation       = static_cast<T>(params.get("bRotationDeg"));
    const T bRp             = static_cast<T>(params.get("bProjectedRange"));
    const T bSigma          = static_cast<T>(params.get("bDepthSigma"));
    const T bSkewness       = static_cast<T>(params.get("bSkewness"));
    const T bKurtosis       = static_cast<T>(params.get("bKurtosis"));
    const T bLatHead        = static_cast<T>(params.get("bLateralSigmaHead"));
    const T bHeadFrac       = static_cast<T>(params.get("bHeadFraction"));
    const T bRpTail         = static_cast<T>(params.get("bTailProjectedRange"));
    const T bSigmaTail      = static_cast<T>(params.get("bTailDepthSigma"));
    const T bSkewnessTail   = static_cast<T>(params.get("bTailSkewness"));
    const T bKurtTail       = static_cast<T>(params.get("bTailKurtosis"));
    const T bLatTail        = static_cast<T>(params.get("bTailLateralSigma"));
    const T bD0             = static_cast<T>(params.get("bAnnealD0"));
    const T bEa             = static_cast<T>(params.get("bAnnealEa"));
    const T bSolC0          = static_cast<T>(params.get("bSolidSolubilityC0"));
    const T bSolEa          = static_cast<T>(params.get("bSolidSolubilityEa"));

    // ── Build substrate ───────────────────────────────────────────────────────
    auto domain = buildSubstrate(xExtent, substrateDepth, topSpace,
                                 padOxideThickness, gridDelta);
    std::cout << "Domain: " << xExtent << " nm wide × " << substrateDepth
              << " nm deep,  Δ = " << gridDelta << " nm\n\n";

    // ──────────────────────────────────────────────────────────────────────────
    // Step 1: P implant — mask the right half (x > 0)
    // ──────────────────────────────────────────────────────────────────────────
    std::cout << "Step 1: P implant (" << pDose << " cm⁻²"
              << "  tilt=" << pTilt << "°  rot=" << pRotation << "°"
              << "  left half)\n";
    auto origMat = maskHalf(domain, /*maskRightHalf=*/true);
    runImplant(domain, "P_total", "P_damage",
               pRp, pSigma, pSkewness, pKurtosis,
               pRpTail, pSigmaTail, pSkewnessTail, pKurtTail,
               pHeadFrac, pDose, pTilt, pLatHead, pLatTail);
    restoreMaterial(domain, origMat);
    std::cout << "  P implant done.\n";

    // ──────────────────────────────────────────────────────────────────────────
    // Step 2: B implant — mask the left half (x < 0)
    // ──────────────────────────────────────────────────────────────────────────
    std::cout << "Step 2: B implant (" << bDose << " cm⁻²"
              << "  tilt=" << bTilt << "°  rot=" << bRotation << "°"
              << "  right half)\n";
    origMat = maskHalf(domain, /*maskRightHalf=*/false);
    runImplant(domain, "B_total", "B_damage",
               bRp, bSigma, bSkewness, bKurtosis,
               bRpTail, bSigmaTail, bSkewnessTail, bKurtTail,
               bHeadFrac, bDose, bTilt, bLatHead, bLatTail);
    restoreMaterial(domain, origMat);
    std::cout << "  B implant done.\n\n";

    // ──────────────────────────────────────────────────────────────────────────
    // Step 3: Zero-time activation — write P_active / B_active without diffusing
    //         (equivalent to Sentaurus "diffuse time=0")
    // ──────────────────────────────────────────────────────────────────────────
    std::cout << "Step 3: Solid activation (no diffusion)\n";

    auto annealP = SmartPointer<Anneal<T, D>>::New();
    annealP->setTemperature(annealTempK);
    annealP->setSpeciesLabel("P_total");
    annealP->setActiveLabel("P_active");
    annealP->setDiffusionMaterials({Material::Si});
    annealP->setBlockingMaterials({Material::Air, Material::SiO2});
    annealP->enableSolidActivation(true);
    annealP->setSolidSolubilityArrhenius(pSolC0, pSolEa);
    annealP->applyActivation(domain);

    auto annealB = SmartPointer<Anneal<T, D>>::New();
    annealB->setTemperature(annealTempK);
    annealB->setSpeciesLabel("B_total");
    annealB->setActiveLabel("B_active");
    annealB->setDiffusionMaterials({Material::Si});
    annealB->setBlockingMaterials({Material::Air, Material::SiO2});
    annealB->enableSolidActivation(true);
    annealB->setSolidSolubilityArrhenius(bSolC0, bSolEa);
    annealB->applyActivation(domain);
    std::cout << "  Activation done.\n\n";

    // ──────────────────────────────────────────────────────────────────────────
    // Step 4: Thermal anneal
    // ──────────────────────────────────────────────────────────────────────────
    std::cout << "Step 4: Anneal " << (annealTempK - 273.15) << " °C / "
              << annealTime << " s\n";

    annealP->setDuration(annealTime);
    annealP->setArrheniusParameters(pD0, pEa);
    annealP->setDamageLabels("P_damage", "P_damage_last");
    Process<T, D>(domain, annealP, T(0)).apply();

    annealB->setDuration(annealTime);
    annealB->setArrheniusParameters(bD0, bEa);
    annealB->setDamageLabels("B_damage", "B_damage_last");
    Process<T, D>(domain, annealB, T(0)).apply();
    std::cout << "  Anneal done.\n\n";

    // ──────────────────────────────────────────────────────────────────────────
    // Step 5: NetDoping — net_doping = P_active − B_active; find junction
    // ──────────────────────────────────────────────────────────────────────────
    std::cout << "Step 5: NetDoping analysis\n";

    viennacs::NetDoping<T, D> nd;
    nd.setCellSet(domain->getCellSet());
    nd.addDonorLabel("P_active");
    nd.addAcceptorLabel("B_active");
    nd.apply();

    const T xj   = nd.lateralJunctionPosition(scanDepth);
    const auto xjs = nd.lateralJunctionPositions(scanDepth);

    std::cout << "  Lateral junction at depth " << scanDepth << " nm:\n";
    if (xj < T(1e30))
        std::cout << "    x_j = " << xj << " nm  (" << xjs.size() << " crossing(s))\n";
    else
        std::cout << "    No lateral junction found at this depth.\n";
    std::cout << "\n";

    // ──────────────────────────────────────────────────────────────────────────
    // Step 6: Sheet resistance of each side
    // ──────────────────────────────────────────────────────────────────────────
    std::cout << "Step 6: Sheet resistance\n";

    viennacs::SheetResistance<T, D> sr;
    sr.setCellSet(domain->getCellSet());

    sr.setConcentrationLabel("P_active");
    std::cout << "  Rsh(n-side, P_active) = " << sr.computeElectron() << " Ω/□\n";

    sr.setConcentrationLabel("B_active");
    std::cout << "  Rsh(p-side, B_active) = " << sr.computeHole() << " Ω/□\n";
    std::cout << "  (Integrals span the whole domain; each side contributes ~half.)\n\n";

    // ──────────────────────────────────────────────────────────────────────────
    // Step 7: Write output files
    // ──────────────────────────────────────────────────────────────────────────
    std::cout << "Step 7: Writing output files\n";

    writeDepthProfile(domain, "P_active",   "pnJunction_P_depth.csv");
    writeDepthProfile(domain, "B_active",   "pnJunction_B_depth.csv");
    writeDepthProfile(domain, "net_doping", "pnJunction_net_depth.csv");
    writeLateralProfile(domain, "net_doping", scanDepth, "pnJunction_lateral.csv");

    domain->getCellSet()->writeVTU("pnJunction_cellset.vtu");
    std::cout << "  Wrote pnJunction_cellset.vtu\n";
    std::cout << "\nDone.\n";
    return 0;
}
