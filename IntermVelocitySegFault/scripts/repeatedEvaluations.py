import viennaps as vps
import viennals as vls
from processSequence import processSequence
from types import SimpleNamespace
from itertools import product

import os
import csv
import time

vps.Logger.setLogLevel(vps.LogLevel.INTERMEDIATE)
# vps.Logger.setLogLevel(vps.LogLevel.DEBUG)

initialDomain = vps.Domain()

dirOfThisScript = os.path.dirname(os.path.abspath(__file__))

gridResolution = 1
# depThickness = 3.1
depThickness = 2.1

vps.Reader(
    initialDomain,
    os.path.join(
        dirOfThisScript,
        f"../data/BP-GT-W2-S3-TCP-1200W-initialDomain-{gridResolution}nm.vpsd",
    ),
).apply()


optimum = {
    "isoEtchDepth": 2.0,
    "depThickness": depThickness,
    "ionFlux": 125.71261101803212,
    "etchantFlux": 5401.405552767991,
    "meanEnergy": 387.0,
    "ionExponent": 200.0,
    "siAie": 5.915203825312938,
    "beta_F": 0.0001,
    "kSigma": 1000.0,
}

numberOfEvaluations = 1

fluxEngineTypeToUse = vps.FluxEngineType.GPU_TRIANGLE
# fluxEngineTypeToUse = vps.FluxEngineType.CPU_DISK
rppValues = [1000]
smoothingNeighborsValues = [2]
# smoothingNeighborsValues = [1, 2]
timeStepRatioValues = [0.4]
# timeStepRatioValues = [0.4999, 0.375, 0.25, 0.1]
# adaptiveTimeStepping = [True, False]
adaptiveTimeStepping = [True]
# rngSeed = [12345]
spatialSchemes = [
    # vps.SpatialScheme.ENGQUIST_OSHER_1ST_ORDER,
    vps.SpatialScheme.LAX_FRIEDRICHS_1ST_ORDER,
    # vps.SpatialScheme.LOCAL_LAX_FRIEDRICHS_1ST_ORDER,
]

temporalSchemes = [
    # vps.TemporalScheme.FORWARD_EULER,
    vps.TemporalScheme.RUNGE_KUTTA_2ND_ORDER,
    # vps.TemporalScheme.RUNGE_KUTTA_3RD_ORDER,
]
intermediateVelocityCalculations = [True]


dirToSaveSurfaceMeshes = f"{dirOfThisScript}/../{fluxEngineTypeToUse.name}"
os.makedirs(dirToSaveSurfaceMeshes, exist_ok=True)

# Create CSV file for logging evaluation times
csv_filename = os.path.join(dirToSaveSurfaceMeshes, "evaluation_times.csv")
csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Surface Mesh Filename", "Evaluation Time (seconds)"])

# Generate all combinations
parameterCombinations = list(
    product(
        rppValues,
        smoothingNeighborsValues,
        timeStepRatioValues,
        adaptiveTimeStepping,
        spatialSchemes,
        temporalSchemes,
        intermediateVelocityCalculations,
    )
)
totalCombinations = len(parameterCombinations)

for configIndex, (
    rppValue,
    smoothingNeighborsValue,
    timeStepRatioValue,
    adaptiveTimeSteppingValue,
    spatialSchemeValue,
    temporalSchemeValue,
    intermediateVelocityCalculationsValue,
) in enumerate(parameterCombinations, 1):
    print(f"\n{'='*80}")
    print(
        f"Configuration {configIndex}/{totalCombinations}: "
        f"rpp={rppValue}, "
        f"smoothingNeighbors={smoothingNeighborsValue}, "
        f"timeStepRatio={timeStepRatioValue}, "
        f"adaptiveTimeStepping={adaptiveTimeSteppingValue}, "
        f"spatialScheme={spatialSchemeValue.name}, "
        f"temporalScheme={temporalSchemeValue.name}, "
        f"intermediateVelocityCalculations="
        f"{intermediateVelocityCalculationsValue}"
    )
    print(f"{'='*80}\n")

    options = SimpleNamespace(
        rpp=rppValue,
        fluxEngineType=fluxEngineTypeToUse,
        smoothingNeighbors=smoothingNeighborsValue,
        timeStepRatio=timeStepRatioValue,
        adaptiveTimeStepping=adaptiveTimeSteppingValue,
        spatialScheme=spatialSchemeValue,
        temporalScheme=temporalSchemeValue,
        intermediateVelocityCalculations=intermediateVelocityCalculationsValue,
        # rngSeed=rngSeedValue
    )

    # Create subdirectory for this configuration
    configDir = (
        f"{dirToSaveSurfaceMeshes}/"
        f"rpp{rppValue}"
        f"-sn{smoothingNeighborsValue}"
        f"-tsr{timeStepRatioValue}"
        f"-ada{adaptiveTimeSteppingValue}"
        f"-int{spatialSchemeValue.name}"
        f"-temp{temporalSchemeValue.name}"
        f"-intVel{intermediateVelocityCalculationsValue}"
    )
    os.makedirs(configDir, exist_ok=True)

    for i in range(numberOfEvaluations):
        print(f"  Evaluation {i + 1}/{numberOfEvaluations}")

        processedDomain = vps.Domain()
        processedDomain.deepCopy(initialDomain)

        # Start timing
        start_time = time.time()

        resultLS, resultDomain = processSequence(
            processedDomain, params=optimum, options=options
        )

        # End timing
        evaluation_time = time.time() - start_time

        filename = (
            f"r{rppValue}"
            f"-s{smoothingNeighborsValue}"
            f"-t{timeStepRatioValue}"
            f"-ada{adaptiveTimeSteppingValue}"
            f"-int{spatialSchemeValue.name}"
            f"-temp{temporalSchemeValue.name}"
            f"-intVel{intermediateVelocityCalculationsValue}"
            f"--eval{i + 1:03d}.vtp"
        )
        resultDomain.saveSurfaceMesh(f"{configDir}/{filename}", True)

        # Log the evaluation time to CSV
        csv_writer.writerow([filename, f"{evaluation_time:.6f}"])
        csv_file.flush()  # Ensure data is written immediately

        print(f"    Evaluation time: {evaluation_time:.3f} seconds")

# Close the CSV file
csv_file.close()
print(f"\n{'='*80}")
print(f"All evaluations complete. Timing data saved to: {csv_filename}")
print(f"{'='*80}")
