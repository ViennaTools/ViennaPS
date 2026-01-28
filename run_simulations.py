"""ViennaPS Basic Simulations
Demonstrates 2D etching and deposition processes.

Note: Uses single-threaded mode for macOS ARM64 compatibility.
"""
import viennaps as ps
from pathlib import Path

# macOS ARM64 compatibility: use single thread
ps.setNumThreads(1)
ps.Logger.setLogLevel(ps.LogLevel.INFO)
ps.setDimension(2)

print("=" * 60)
print("ViennaPS Basic Simulations Demo (2D)")
print("=" * 60)

# Create output directory
output_dir = Path("simulation_output")
output_dir.mkdir(exist_ok=True)

# ============================================================
# Simulation 1: Isotropic Etching on Plane
# ============================================================
print("\n[1] Isotropic Etching on Plane...", flush=True)

# Create domain
domain1 = ps.Domain(1.0, 20.0, 15.0)
print("  Domain: 20x15 um, grid delta = 1.0 um", flush=True)

# Create plane at y=0
ps.MakePlane(domain1, 0.0, ps.Material.Si).apply()
print("  Plane geometry created (Si substrate)", flush=True)

# Save initial
domain1.saveSurfaceMesh(str(output_dir / "01_plane_initial.vtp"))

# Isotropic etching (negative rate = etching)
model1 = ps.IsotropicProcess(rate=-0.5)
process1 = ps.Process(domain1, model1, 8.0)
print("  Running isotropic etch (rate=-0.5, time=8)...", flush=True)
process1.apply()
print("  Etching complete!", flush=True)

domain1.saveSurfaceMesh(str(output_dir / "01_plane_etched.vtp"))
print("  Saved: 01_plane_initial.vtp, 01_plane_etched.vtp", flush=True)

# ============================================================
# Simulation 2: Trench with Single Particle Deposition
# ============================================================
print("\n[2] Trench Deposition (Single Particle)...", flush=True)

# Create domain and trench
domain2 = ps.Domain(0.5, 20.0, 20.0)
ps.MakeTrench(
    domain=domain2,
    trenchWidth=8.0,
    trenchDepth=10.0,
    trenchTaperAngle=0.0,
).apply()
print("  Trench: 8um wide x 10um deep", flush=True)

# Save initial
domain2.saveSurfaceMesh(str(output_dir / "02_trench_initial.vtp"))

# Duplicate top level set for deposition material
domain2.duplicateTopLevelSet(ps.Material.SiO2)

# Single particle deposition
model2 = ps.SingleParticleProcess(
    rate=1.0,
    stickingProbability=0.1,  # Low = more conformal
    sourceExponent=1.0,
    maskMaterial=ps.Material.Mask
)
process2 = ps.Process(domain2, model2, 15.0)
print("  Running single particle deposition (sticking=0.1, time=15)...", flush=True)
process2.apply()
print("  Deposition complete!", flush=True)

domain2.saveSurfaceMesh(str(output_dir / "02_trench_deposited.vtp"))
print("  Saved: 02_trench_initial.vtp, 02_trench_deposited.vtp", flush=True)

# ============================================================
# Simulation 3: Directional Etching
# ============================================================
print("\n[3] Directional Etching...", flush=True)

# Create domain with trench (using mask)
domain3 = ps.Domain(0.5, 25.0, 20.0)
ps.MakeTrench(
    domain=domain3,
    trenchWidth=6.0,
    trenchDepth=2.0,  # Shallow starting trench
    trenchTaperAngle=0.0,
    maskHeight=2.0,  # Mask on top
).apply()
print("  Shallow trench with mask: 6um wide, 2um deep", flush=True)

domain3.saveSurfaceMesh(str(output_dir / "03_directional_initial.vtp"))

# Directional etching (anisotropic)
# Note: direction needs 3 components even in 2D, params use "Velocity" not "Rate"
model3 = ps.DirectionalProcess(
    direction=[0.0, -1.0, 0.0],  # Downward (Y-negative)
    directionalVelocity=-2.0,
    isotropicVelocity=-0.1,
    maskMaterial=ps.Material.Mask
)
process3 = ps.Process(domain3, model3, 6.0)
print("  Running directional etch (dir=-Y, time=6)...", flush=True)
process3.apply()
print("  Etching complete!", flush=True)

domain3.saveSurfaceMesh(str(output_dir / "03_directional_etched.vtp"))
print("  Saved: 03_directional_initial.vtp, 03_directional_etched.vtp", flush=True)

# ============================================================
# Simulation 4: Geometric Deposition (Sphere Distribution)
# ============================================================
print("\n[4] Geometric Deposition (Conformal Layer)...", flush=True)

# Create domain with trench
domain4 = ps.Domain(0.5, 20.0, 15.0)
ps.MakeTrench(
    domain=domain4,
    trenchWidth=5.0,
    trenchDepth=8.0,
    trenchTaperAngle=0.0,
).apply()
print("  Trench: 5um wide x 8um deep", flush=True)

domain4.saveSurfaceMesh(str(output_dir / "04_geometric_initial.vtp"))

# Duplicate for deposition
domain4.duplicateTopLevelSet(ps.Material.SiO2)

# Geometric deposition (perfectly conformal layer)
model4 = ps.SphereDistribution(radius=1.5)  # 1.5um conformal layer
process4 = ps.Process(domain4, model4, 0.0)  # Time doesn't matter for geometric
print("  Applying conformal layer (1.5um thick)...", flush=True)
process4.apply()
print("  Deposition complete!", flush=True)

domain4.saveSurfaceMesh(str(output_dir / "04_geometric_deposited.vtp"))
print("  Saved: 04_geometric_initial.vtp, 04_geometric_deposited.vtp", flush=True)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("All simulations complete!")
print("=" * 60)
print("\nOutput files in ./simulation_output/:")
vtp_files = sorted(output_dir.glob("*.vtp"))
for f in vtp_files:
    print(f"  - {f.name}")
print(f"\nTotal: {len(vtp_files)} VTP files")
print("\nView results with ParaView or similar VTK viewer.")
