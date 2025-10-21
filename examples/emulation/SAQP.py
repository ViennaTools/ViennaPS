"""
Self-Aligned Quadruple Patterning (SAQP) Process

This script demonstrates a complete SAQP process flow which is used in advanced
semiconductor manufacturing to achieve 4x pattern density multiplication.

Process Flow:
1. Create initial mandrels (core pattern) on substrate
2. Deposit first spacer material (conformal deposition)
3. Anisotropic etch back to form sidewall spacers
4. Remove mandrels (selective etch)
5. Deposit second spacer material on first spacers
6. Anisotropic etch back to form second generation spacers
7. Pattern transfer: etch into substrate using spacers as mask
8. Remove remaining spacer materials

Result: 4 lines from 1 initial mandrel
"""

import viennaps as ps

# Configuration
ps.Logger.setLogLevel(ps.LogLevel.WARNING)

# Set dimension
ps.setDimension(2)

# Process parameters (all in nanometers)
gridDelta = 0.35
xExtent = 100.0

# Mandrel parameters
mandrelWidth = 15.0
mandrelHeight = 40.0
mandrelPitch = 50.0  # Center-to-center distance

# Spacer parameters
spacerThickness = 9.0  # Target spacer width
spacer2Thickness = 4.0  # Second spacer thickness
depositionStickyProbability = 0.1

# Etch parameters
substrateEtchDepth = 70.0

# Output control
outputCounter = 0


def saveSurface(domain, label=""):
    """Save surface mesh with sequential numbering"""
    global outputCounter
    filename = f"SAQP_{outputCounter:02d}"
    domain.saveSurfaceMesh(filename + ".vtp", True)
    domain.saveVolumeMesh(filename, True)
    print(f"Saved: {filename}")
    outputCounter += 1


# Initialize domain
print("\n" + "=" * 60)
print("SELF-ALIGNED QUADRUPLE PATTERNING (SAQP) SIMULATION")
print("=" * 60 + "\n")

bounds = [0.0, xExtent, -1.0, 1.0]
boundaryConditions = [
    ps.BoundaryType.PERIODIC_BOUNDARY,
    ps.BoundaryType.INFINITE_BOUNDARY,
]

domain = ps.Domain(
    bounds=bounds, gridDelta=gridDelta, boundaryConditions=boundaryConditions
)

# Step 1: Create substrate
print("Step 1: Creating silicon substrate...")
ps.MakePlane(domain, 0.0, ps.Material.Si).apply()
saveSurface(domain, "substrate")

# Step 2: Create mandrels (initial core pattern)
print("\nStep 2: Creating mandrels (core pattern)...")
# Create two mandrels to show the pattern multiplication
# First mandrel at x = 25nm
box = ps.ls.Domain(domain.getGrid())
ps.ls.MakeGeometry(
    box,
    ps.ls.Box([25.0 - mandrelWidth / 2, 0.0], [25.0 + mandrelWidth / 2, mandrelHeight]),
).apply()

box2 = ps.ls.Domain(domain.getGrid())
ps.ls.MakeGeometry(
    box2,
    ps.ls.Box(
        [25.0 + mandrelPitch - mandrelWidth / 2, 0.0],
        [25.0 + mandrelPitch + mandrelWidth / 2, mandrelHeight],
    ),
).apply()
ps.ls.BooleanOperation(box, box2, ps.ls.BooleanOperationEnum.UNION).apply()
domain.insertNextLevelSetAsMaterial(box, ps.Material.SiN)


# Shift the trench position by creating another one
# We'll use a mask approach for multiple mandrels
print("  Mandrel width: {} nm".format(mandrelWidth))
print("  Mandrel height: {} nm".format(mandrelHeight))
saveSurface(domain, "mandrels")

# Step 3: Deposit first spacer material (conformal coating)
print("\nStep 3: Depositing first spacer material (SiO2)...")
print("  Spacer thickness: {} nm".format(spacerThickness))

# Duplicate top level set for deposition
domain.duplicateTopLevelSet(ps.Material.SiO2)

# Use single particle process for conformal deposition
model_spacer1_dep = ps.SingleParticleProcess(
    rate=1.0,
    stickingProbability=depositionStickyProbability,
    sourceExponent=1.0,  # Cosine distribution
)

process_spacer1 = ps.Process(domain, model_spacer1_dep, spacerThickness)
process_spacer1.apply()
print("  First spacer deposited")
saveSurface(domain, "spacer1_deposited")

# Step 4: Anisotropic etch back to form sidewall spacers
print("\nStep 4: Anisotropic etch back to form first spacers...")
# Directional etch that removes horizontal surfaces but leaves vertical spacers
model_etch_spacer1 = ps.DirectionalProcess(
    direction=[0.0, 1.0, 0.0],
    directionalVelocity=1.0,
    isotropicVelocity=0.05,  # Small isotropic component
    maskMaterial=ps.Material.Si,  # Protect substrate
    calculateVisibility=False,
)

# Etch back slightly more than deposited to ensure only sidewall spacers remain
process_etch1 = ps.Process(domain, model_etch_spacer1, spacerThickness + 5.0)
process_etch1.apply()
print("  First spacers formed (sidewalls only)")
saveSurface(domain, "spacer1_etched")

# Step 5: Remove mandrels (selective etch)
print("\nStep 5: Removing mandrels (selective etch)...")
# Use isotropic etch that only etches SiN (mandrel material)
domain.removeMaterial(ps.Material.SiN)
print("  Mandrels removed - first spacers remain")
saveSurface(domain, "mandrels_removed")

# Step 6: Deposit second spacer material (on first spacers)
print("\nStep 6: Depositing second spacer material (SiN)...")
print("  Second spacer thickness: {} nm".format(spacer2Thickness))

domain.duplicateTopLevelSet(ps.Material.SiN)

model_spacer2_dep = ps.SingleParticleProcess(
    rate=1.0, stickingProbability=depositionStickyProbability, sourceExponent=1.0
)

process_spacer2 = ps.Process(domain, model_spacer2_dep, spacer2Thickness)
process_spacer2.apply()
print("  Second spacer deposited")
saveSurface(domain, "spacer2_deposited")

# Step 7: Anisotropic etch back to form second generation spacers
print("\nStep 7: Anisotropic etch back to form second spacers...")
model_etch_spacer2 = ps.DirectionalProcess(
    direction=[0.0, 1.0, 0.0],
    directionalVelocity=1.0,
    isotropicVelocity=0.05,
    maskMaterial=ps.Material.Si,
    calculateVisibility=False,
)

process_etch2 = ps.Process(domain, model_etch_spacer2, spacer2Thickness + 3.0)
process_etch2.apply()
print("  Second spacers formed")
print("  Pattern multiplication achieved: 1 mandrel -> 4 lines")
saveSurface(domain, "spacer2_etched")

# Step 8: Pattern transfer - etch into substrate
print("\nStep 8: Pattern transfer - etching substrate...")
print("  Etch depth: {} nm".format(substrateEtchDepth))

# Use directional etch to transfer pattern to silicon substrate
# Both SiO2 and SiN act as etch masks
model_pattern_transfer = ps.SingleParticleProcess(
    rate=-1.0,
    stickingProbability=1.0,
    sourceExponent=1000,
    maskMaterials=[ps.Material.SiO2, ps.Material.SiN],
)

process_transfer = ps.Process(domain, model_pattern_transfer, substrateEtchDepth)
process_transfer.apply()
print("  Pattern transferred to substrate")
saveSurface(domain, "pattern_transferred")

# Step 9: Remove spacer materials (optional cleanup)
print("\nStep 9: Removing spacer materials...")
# Remove SiN spacers
domain.removeMaterial(ps.Material.SiN)
# Remove SiO2 spacers
domain.removeMaterial(ps.Material.SiO2)
print("  Spacers removed - final pattern revealed")
saveSurface(domain, "final_pattern")

# Summary
print("\n" + "=" * 60)
print("SAQP PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)
print("\nProcess Summary:")
print(f"  - Initial mandrel width: {mandrelWidth} nm")
print(f"  - First spacer thickness: {spacerThickness} nm")
print(f"  - Second spacer thickness: {spacer2Thickness} nm")
print(f"  - Final etch depth: {substrateEtchDepth} nm")
print(f"  - Pattern multiplication: 1 mandrel -> 4 features")
print(f"  - Grid resolution: {gridDelta} nm")
print(f"\nTotal output files: {outputCounter}")
print("\nCheck the generated .vtp files to visualize each step!")
print("=" * 60 + "\n")
