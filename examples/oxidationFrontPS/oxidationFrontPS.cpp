#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <lsAdvect.hpp>
#include <psDomain.hpp>
#include <process/psProcess.hpp>
#include <process/psTranslationField.hpp>

// #include <psUtils.hpp>

// ViennaCS includes are available since ViennaPS links against it
#include <csDenseCellSet.hpp>

#include "geometryPS.hpp"

namespace ps = viennaps;
namespace cs = viennacs;
namespace ls = viennals;

using NumericType = double;
constexpr int D = 3;

// Material IDs
constexpr ps::Material MAT_SUBSTRATE = ps::Material::Si;
constexpr ps::Material MAT_OXIDE = ps::Material::SiO2; // We will use this for the oxide layer
constexpr ps::Material MAT_MASK = ps::Material::Mask;
constexpr ps::Material MAT_AMBIENT = ps::Material::GAS;
constexpr ps::Material MAT_VOID = ps::Material::Undefined;

class OxidationSimulation {
public:
  OxidationSimulation(const std::string &configFile) {
    params.readConfigFile(configFile);
    omp_set_num_threads(params.get<int>("numThreads"));
    
    ps::Logger::setLogLevel(ps::LogLevel::DEBUG);

    // Cache parameters
    k_base = params.get("reactionRateConstant");
    alpha = params.get("eFieldInfluence");
    D_ox = params.get("oxidantDiffusivity");
    ambientOxidant = params.get("ambientOxidant");
    dx = params.get("gridDelta");
    duration = params.get("duration");
    timeStabilityFactor = params.get("timeStabilityFactor");
    try {
      topologyRebuildInterval = params.get<int>("topologyRebuildInterval");
    } catch (...) {
      topologyRebuildInterval = 5;
    }

    // Initialize Domain and Geometry
    initGeometry();
  }

  void apply() {
    NumericType time = 0.;
    int step = 0;
    bool materialChanged = true;

    // Initial CellSet generation
    std::cout << "Generating initial CellSet..." << std::endl;
    generateCellSet();
    precomputeEFieldIndices();
    initOxidationFields();
    
    // Write initial state
    domain->getCellSet()->writeVTU("oxidation_initial.vtu");

    while (time < duration) {
      std::cout << "--- Step " << step << " (Time: " << time << ") ---" << std::endl;
      
      if (step % topologyRebuildInterval == 0 || materialChanged) {
        updateActiveCells();
        materialChanged = false;
      }

      loadElectricField();
      // Calculate max reaction rate for time step control
      calculateReactionRates();
      NumericType max_k = calculateMaxReactionRate();

      // 1. Determine Time Step
      NumericType dt_diff = dx * dx / (D_ox * 2 * D);
      NumericType dt_react = (max_k > 1e-12) ? (1.0 / max_k) : duration;
      // Use dt_react as limit for interface advection stability
      NumericType dt = std::min(dt_diff, dt_react);// * timeStabilityFactor;

      if (time + dt > duration) dt = duration - time;
      if (step == 0) std::cout << "Initial dt: " << dt << std::endl;

      // 2. Solve Diffusion (Explicit)
      std::cout << "Solving diffusion (Explicit)..." << std::endl;
      solveDiffusionExplicit(dt);

      // 3. Update Oxide Fraction (ViennaCS logic)
      // This evolves the materials in the CellSet
      if (updateOxideFraction(dt)) {
        materialChanged = true;
      }

      // 4. Advect Interface (ViennaPS feature)
      // Calculate velocity v = -k * C_ox at the interface
      // We store this velocity on the CellSet, ViennaPS will map it to the Level Set
      calculateVelocity();
      std::cout << "Advecting interface..." << std::endl;

      {
        auto velocityField = ps::SmartPointer<CustomVelocityField>::New(cellSet, "velocity");
        auto translationField = ps::SmartPointer<ps::TranslationField<NumericType, D>>::New(
            velocityField, domain->getMaterialMap(), 0);
        ls::Advect<NumericType, D> advectionKernel;
        advectionKernel.insertNextLevelSet(domain->getLevelSets()[0]);
        advectionKernel.setVelocityField(translationField);
        advectionKernel.setAdvectionTime(dt);
        advectionKernel.apply();
      }

      time += dt;
      step++;

      if (step % 10 == 0) {
        std::cout << "Step " << step << " Time: " << time << " dt: " << dt << std::endl;
        cellSet->writeVTU("oxidation_step_" + std::to_string(step) + ".vtu");

        auto mesh = ls::SmartPointer<ls::Mesh<NumericType>>::New();
        ls::ToSurfaceMesh<NumericType, D>(domain->getLevelSets()[0], mesh).apply();
        ls::VTKWriter<NumericType>(mesh, "oxidation_ls_step_" + std::to_string(step) + ".vtp").apply();
      }
    }

    cellSet->writeVTU("oxidation_final.vtu");
    std::cout << "Simulation Done." << std::endl;
  }

private:
  ps::util::Parameters params;
  ps::SmartPointer<ps::Domain<NumericType, D>> domain;
  ps::SmartPointer<cs::DenseCellSet<NumericType, D>> cellSet;

  class CustomVelocityField : public ps::VelocityField<NumericType, D> {
    ps::SmartPointer<cs::DenseCellSet<NumericType, D>> cellSet;
    std::string velocityName;
    std::vector<NumericType> *velocityData = nullptr;

  public:
    CustomVelocityField(ps::SmartPointer<cs::DenseCellSet<NumericType, D>> cs,
                        std::string name)
        : cellSet(cs), velocityName(name) {
        velocityData = cellSet->getScalarData(velocityName);
    }

    NumericType getScalarVelocity(const viennacore::Vec3D<NumericType> &coord, int,
                                  const viennacore::Vec3D<NumericType> &,
                                  unsigned long) override {
      int idx = cellSet->getIndex(coord);
      if (idx >= 0) {
        if (velocityData && idx < velocityData->size())
          return (*velocityData)[idx];
      }
      return 0.0;
    }

    viennacore::Vec3D<NumericType> getVectorVelocity(const viennacore::Vec3D<NumericType> &, int,
                                                 const viennacore::Vec3D<NumericType> &,
                                                 unsigned long) override {
      return {0., 0., 0.};
    }
  };

  NumericType k_base, alpha, D_ox, ambientOxidant, dx, duration, timeStabilityFactor;
  std::vector<NumericType> electricField1D;
  std::vector<int> efieldIndices;
  std::vector<NumericType> reactionRates;
  std::vector<NumericType> nextOxidant;
  std::vector<int> activeCells;
  std::vector<char> isActiveCell;
  int topologyRebuildInterval;

  void initGeometry() {
    // Create Domain
    domain = ps::SmartPointer<ps::Domain<NumericType, D>>::New(
        params.get("gridDelta"), params.get("xExtent"), params.get("yExtent"));

    // Use geometryPS.hpp to match ViennaCS example
    auto matMap = ls::SmartPointer<ls::MaterialMap>::New();
    std::vector<int> materials;
    auto levelSets = geometry::makeStructure<NumericType, D>(
        params, matMap, materials, static_cast<int>(MAT_SUBSTRATE),
        static_cast<int>(MAT_MASK), static_cast<int>(MAT_AMBIENT));

    for (size_t i = 0; i < levelSets.size(); ++i) {
      domain->insertNextLevelSetAsMaterial(levelSets[i],
                                           static_cast<ps::Material>(materials[i]));
    }
  }

  void generateCellSet() {
    NumericType depth = params.get("substrateHeight") + params.get("maskHeight") + params.get("ambientHeight");
    // Generate CellSet. "true" indicates we want cells above the surface (gas), 
    // but here we want the whole domain to solve diffusion.
    domain->generateCellSet(depth, MAT_VOID, true);
    cellSet = std::dynamic_pointer_cast<cs::DenseCellSet<NumericType, D>>(domain->getCellSet());
    cellSet->buildNeighborhood();
  }

  void loadElectricField() {
    electricField1D.clear();
    std::string csvFileName = params.get<std::string>("EfieldFile");

    std::ifstream in(csvFileName);
    if (!in.good()) {
        std::cerr << "Error: cannot open file " << csvFileName << std::endl;
        return;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::stringstream ss(line);
        std::string segment;
        std::vector<NumericType> row;
        while (std::getline(ss, segment, ',')) {
            try { row.push_back(std::stod(segment)); } catch (...) {}
        }
        
        // Match oxidationFront.cpp parsing logic
        if (D == 2 && row.size() >= 2) {
            electricField1D.push_back(row[1]);
        } else if (D == 3 && row.size() >= 3) {
            electricField1D.push_back(row[2]);
        } else if (!row.empty()) {
            electricField1D.push_back(row.back());
        }
    }
  }

  void precomputeEFieldIndices() {
    efieldIndices.resize(cellSet->getNumberOfCells());
    // Match oxidationFront.cpp mapping constants
    const NumericType csvDx = 1.0;
    const NumericType csvExtent = 150.0;
    const int nx = static_cast<int>(csvExtent / csvDx);

    #pragma omp parallel for
    for (int i = 0; i < cellSet->getNumberOfCells(); ++i) {
        auto center = cellSet->getCellCenter(i);
        // Map coordinates: Sim X [-x/2, x/2] -> Gen X [0, x] using csvExtent
        NumericType gen_x = center[0] + (csvExtent / 2.0);

        int ix = static_cast<int>(gen_x / csvDx);
        int idx = -1;

        if constexpr (D == 2) {
            idx = ix;
        } else {
            NumericType gen_y = center[1] + (csvExtent / 2.0);
            int iy = static_cast<int>(gen_y / csvDx);
            if (iy >= 0 && iy < nx) idx = iy * nx + ix;
        }

        efieldIndices[i] = idx;
    }
  }

  void initOxidationFields() {
    auto oxidant = cellSet->addScalarData("oxidant", 0.0);
    auto oxideFraction = cellSet->addScalarData("oxideFraction", 0.0);
    auto materials = cellSet->getScalarData("Material");
    
    if (!materials) {
        std::cerr << "Error: Material field missing in initOxidationFields!" << std::endl;
        return;
    }

    // Initialize oxideFraction based on Material ID to drive diffusion
    // In ViennaPS, the geometry is explicit, so we set fraction=1 in Oxide, 0 in Substrate
    #pragma omp parallel for
    for (int i = 0; i < cellSet->getNumberOfCells(); ++i) {
      int mat = static_cast<int>((*materials)[i]);
      if (mat == static_cast<int>(MAT_AMBIENT)) {
        (*oxidant)[i] = ambientOxidant;
        (*oxideFraction)[i] = 0.0; // Gas
      } else if (mat == static_cast<int>(MAT_OXIDE)) {
        (*oxideFraction)[i] = 1.0;
      } else {
        (*oxideFraction)[i] = 0.0;
      }
    }
  }

  NumericType getReactionRate(NumericType E_mag, NumericType oxideFrac) {
    NumericType availableSi = std::max(0.0, 1.0 - oxideFrac);
    return k_base * (1.0 + alpha * std::abs(E_mag)) * availableSi;
  }

  NumericType getDiffusivity(int material, NumericType oxideFrac) {
    if (material == static_cast<int>(MAT_MASK)) return 0.0;
    if (material == static_cast<int>(MAT_AMBIENT)) return 0.0; // Dirichlet BC handled separately
    return D_ox * oxideFrac;
  }

  void calculateReactionRates() {
      if (reactionRates.size() != cellSet->getNumberOfCells()) {
          reactionRates.resize(cellSet->getNumberOfCells());
      }
      auto oxideFractions = cellSet->getScalarData("oxideFraction");
      auto materials = cellSet->getScalarData("Material");

      #pragma omp parallel for
      for (int i = 0; i < cellSet->getNumberOfCells(); ++i) {
          int mat = static_cast<int>((*materials)[i]);
          if (mat == static_cast<int>(MAT_SUBSTRATE) || mat == static_cast<int>(MAT_OXIDE)) {
              NumericType E = 0.0;
              int idx = efieldIndices[i];
              if (idx >= 0 && idx < electricField1D.size()) E = electricField1D[idx];
              reactionRates[i] = getReactionRate(E, (*oxideFractions)[i]);
          } else {
              reactionRates[i] = 0.0;
          }
      }
  }

  void updateActiveCells() {
    auto materials = cellSet->getScalarData("Material");
    auto oxideFractions = cellSet->getScalarData("oxideFraction");

    activeCells.clear();
    isActiveCell.assign(cellSet->getNumberOfCells(), 0);

    // Track which cells are "Core Active" (Oxide or Source)
    std::vector<char> isCoreActive(cellSet->getNumberOfCells(), 0);

    #pragma omp parallel for
    for (int i = 0; i < cellSet->getNumberOfCells(); ++i) {
      int mat = static_cast<int>((*materials)[i]);
      if (mat == static_cast<int>(MAT_SUBSTRATE) ||
          mat == static_cast<int>(MAT_OXIDE)) {
        bool active = (*oxideFractions)[i] > 1e-6;
        if (!active) {
          // Check if touching ambient (Source term)
          auto neighbors = cellSet->getNeighbors(i);
          for (auto n : neighbors) {
            if (n >= 0) {
                int mat_n = static_cast<int>((*materials)[n]);
                if (mat_n == static_cast<int>(MAT_AMBIENT)) {
                    active = true;
                    break;
                }
            }
          }
        }
        isCoreActive[i] = active;
      }
    }

    // Expand to neighbors to allow front propagation
    #pragma omp parallel for
    for (int i = 0; i < cellSet->getNumberOfCells(); ++i) {
      int mat = static_cast<int>((*materials)[i]);
      if (mat == static_cast<int>(MAT_SUBSTRATE) ||
          mat == static_cast<int>(MAT_OXIDE)) {
        // Solve if Core Active OR neighbor of Core Active (Narrow Band)
        bool shouldSolve = isCoreActive[i];
        if (!shouldSolve) {
          auto neighbors = cellSet->getNeighbors(i);
          for (auto n : neighbors) {
            if (n >= 0 && isCoreActive[n]) {
              shouldSolve = true;
              break;
            }
          }
        }

        if (shouldSolve) {
          isActiveCell[i] = 1;
        }
      }
    }

    for (int i = 0; i < cellSet->getNumberOfCells(); ++i) {
      if (isActiveCell[i]) activeCells.push_back(i);
    }
  }

  void solveDiffusionExplicit(NumericType dt) {
    auto oxidant = cellSet->getScalarData("oxidant");
    auto materials = cellSet->getScalarData("Material");
    auto oxideFractions = cellSet->getScalarData("oxideFraction");

    if (nextOxidant.size() != cellSet->getNumberOfCells()) {
      nextOxidant.resize(cellSet->getNumberOfCells());
    }

    NumericType dtdx2 = dt / (dx * dx);

    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(activeCells.size()); ++idx) {
        int i = activeCells[idx];
        int mat = static_cast<int>((*materials)[i]);

        NumericType C_old = (*oxidant)[i];
        NumericType k = reactionRates[i];
        NumericType diffusion_term = 0.0;
        NumericType D_center = getDiffusivity(mat, (*oxideFractions)[i]);

        auto neighbors = cellSet->getNeighbors(i);
        for (int n : neighbors) {
          if (n < 0) continue;
          int mat_n = static_cast<int>((*materials)[n]);
          
          if (mat_n == static_cast<int>(MAT_MASK) || mat_n == static_cast<int>(MAT_VOID)) continue;

          if (mat_n == static_cast<int>(MAT_AMBIENT)) {
            // Dirichlet BC
            diffusion_term += D_ox * (ambientOxidant - C_old);
          } else if (isActiveCell[n]) {
             // Neighbor is Substrate or Oxide and Active
             NumericType D_neighbor = getDiffusivity(mat_n, (*oxideFractions)[n]);
             NumericType D_eff = 0.5 * (D_center + D_neighbor);
             diffusion_term += D_eff * ((*oxidant)[n] - C_old);
          }
        }
        
        nextOxidant[i] = C_old + dtdx2 * diffusion_term - dt * k * C_old;
    }

    // Update
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(activeCells.size()); ++idx) {
        int i = activeCells[idx];
        (*oxidant)[i] = nextOxidant[i];
    }
  }

  bool updateOxideFraction(NumericType dt) {
    auto oxidant = cellSet->getScalarData("oxidant");
    auto oxideFractions = cellSet->getScalarData("oxideFraction");
    auto materials = cellSet->getScalarData("Material");
    int materialChanged = 0;

    #pragma omp parallel for reduction(+:materialChanged)
    for (int i = 0; i < cellSet->getNumberOfCells(); ++i) {
      int mat = static_cast<int>((*materials)[i]);
      if (mat == static_cast<int>(MAT_SUBSTRATE) || mat == static_cast<int>(MAT_OXIDE)) {
        NumericType C = (*oxidant)[i];
        if (C > 1e-12) {
           NumericType k = reactionRates[i];
           NumericType dFrac = k * C * dt;
           (*oxideFractions)[i] += dFrac;
           if ((*oxideFractions)[i] > 1.0) (*oxideFractions)[i] = 1.0;
           
           if ((*oxideFractions)[i] > 0.5 && mat == static_cast<int>(MAT_SUBSTRATE)) {
               (*materials)[i] = static_cast<int>(MAT_OXIDE);
               materialChanged = 1;
           }
        }
      }
    }
    return materialChanged > 0;
  }

  void calculateVelocity() {
    auto velocity = cellSet->addScalarData("velocity", 0.0);
    auto oxidant = cellSet->getScalarData("oxidant");
    auto materials = cellSet->getScalarData("Material");

    if (!oxidant || !materials || !velocity) return;

    #pragma omp parallel for
    for (int i = 0; i < cellSet->getNumberOfCells(); ++i) {
      int mat = static_cast<int>((*materials)[i]);
      // Calculate velocity in both Substrate and Oxide to ensure robust tracking
      if (mat == static_cast<int>(MAT_SUBSTRATE) || mat == static_cast<int>(MAT_OXIDE)) {
          NumericType E = 0.0;
          int idx = efieldIndices[i];
          if (idx >= 0 && idx < electricField1D.size()) E = electricField1D[idx];
          
          // Use potential rate (assuming full Si availability) for LS velocity.
          // This prevents the LS from stopping if it lags into the fully oxidized region.
          NumericType k_potential = k_base * (1.0 + alpha * std::abs(E));
          
          // v = -k * C * dx (Negative because the interface moves into the substrate)
          (*velocity)[i] = -k_potential * (*oxidant)[i] * dx;
      } else {
          (*velocity)[i] = 0.0;
      }
    }

    // Extend velocity into the substrate to ensure the Level Set interface (which lies between
    // active and inactive cells) sees the peak reaction velocity, preventing lag.
    std::vector<NumericType> originalVelocity = *velocity;
    #pragma omp parallel for
    for (int i = 0; i < cellSet->getNumberOfCells(); ++i) {
        int mat = static_cast<int>((*materials)[i]);
        if (mat == static_cast<int>(MAT_SUBSTRATE)) {
            NumericType max_speed = std::abs(originalVelocity[i]);
            for (int n : cellSet->getNeighbors(i)) { // Check all neighbors
                if (n >= 0) {
                    max_speed = std::max(max_speed, std::abs(originalVelocity[n]));
                }
            }
            (*velocity)[i] = -max_speed;
        }
    }
  }

  NumericType calculateMaxReactionRate() {
    NumericType max_k = 0.0;
    
    #pragma omp parallel for reduction(max:max_k)
    for (int i = 0; i < reactionRates.size(); ++i) {
        if (reactionRates[i] > max_k) max_k = reactionRates[i];
    }
    return max_k;
  }
};

int main(int argc, char **argv) {
  std::string configFile = "config.txt";
  if (argc > 1) configFile = argv[1];
  
  OxidationSimulation sim(configFile);
  sim.apply();
  return 0;
}